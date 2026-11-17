# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Inventory a live HF/PyTorch model and replace explicitly selected standard leaves.

The CLI only inventories a trusted local model factory. Programmatic replacement
requires materialized CUDA parameters and does not enable low precision.
"""

# Exact types exclude subclass-specific tensor and forward semantics.
# pylint: disable=unidiomatic-typecheck

import argparse
from collections import Counter, defaultdict
import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn


def _shared_parameters(model):
    """Find parameter identities and storage shared by different qualified names."""
    identities = defaultdict(list)
    storage = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        identities[id(parameter)].append(name)
        if (
            type(parameter) is nn.Parameter
            and not parameter.is_meta
            and parameter.layout == torch.strided
            and parameter.numel()
        ):
            storage[(parameter.device, parameter.untyped_storage().data_ptr())].append(name)
    return {
        name
        for groups in (identities, storage)
        for names in groups.values()
        if len(names) > 1
        for name in names
    }


def _reason(module, path, aliases, shared_parameters):
    """Return why a leaf needs an adapter, or None for a mechanical candidate."""
    if type(module).__module__.startswith("transformer_engine.pytorch."):
        return "already a Transformer Engine module"
    if type(module) not in (nn.Linear, nn.LayerNorm):
        return "custom or unsupported module; inspect for an architecture adapter"
    if module.weight is None:
        return "missing weight parameter"
    if aliases[id(module)] > 1:
        return "shared module has multiple paths"
    if "forward" in vars(module) or any(
        value for key, value in vars(module).items() if key.endswith("_hooks")
    ):
        return "custom forward or hooks"
    if hasattr(module, "_hf_hook") or list(module.buffers()) or list(module.children()):
        return "dispatch hook, buffers, or child modules"
    if type(module) is nn.LayerNorm and (
        len(module.normalized_shape) != 1 or module.weight is None or module.bias is None
    ):
        return "LayerNorm requires one normalized axis and both affine parameters"
    expected_shape = (
        (module.out_features, module.in_features)
        if type(module) is nn.Linear
        else tuple(module.normalized_shape)
    )
    if tuple(module.weight.shape) != expected_shape or (
        module.bias is not None and tuple(module.bias.shape) != (expected_shape[0],)
    ):
        return "parameter shapes disagree with module dimensions"
    if set(dict(module.named_parameters(recurse=False))) - {"weight", "bias"}:
        return "additional parameters"
    for name, parameter in module.named_parameters(recurse=False):
        qualified = f"{path}.{name}" if path else name
        if qualified in shared_parameters:
            return "shared parameter identity or storage requires explicit tie handling"
        if (
            type(parameter) is not nn.Parameter
            or parameter.is_meta
            or parameter.layout != torch.strided
        ):
            return "nonstandard or unmaterialized parameter"
        if parameter.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return "unsupported native parameter dtype"
        if not parameter.is_contiguous() or not parameter.numel():
            return "noncontiguous or empty parameter"
        if getattr(parameter, "_backward_hooks", None) or getattr(
            parameter, "_post_accumulate_grad_hooks", None
        ):
            return "parameter gradient hooks"
    if module.bias is not None and (
        module.bias.dtype != module.weight.dtype or module.bias.device != module.weight.device
    ):
        return "mixed parameter dtype or device"
    return None


def inventory(model):
    """Describe all leaves without importing TE, editing modules, or running forward."""
    modules = list(model.named_modules(remove_duplicate=False))
    aliases = Counter(id(module) for _, module in modules)
    shared = _shared_parameters(model)
    blocked = any(
        getattr(model, name, None)
        for name in (
            "hf_device_map",
            "is_quantized",
            "is_loaded_in_8bit",
            "is_loaded_in_4bit",
            "peft_config",
        )
    ) or any(hasattr(module, "_hf_hook") for _, module in modules)
    entries = []
    for path, module in modules:
        if list(module.children()):
            continue
        reason = _reason(module, path, aliases, shared)
        if blocked:
            reason = "dispatched, quantized, or PEFT model needs construction-time integration"
        if not path:
            reason = "root replacement requires an explicit wrapper"
        entries.append(
            {
                "path": path,
                "type": f"{type(module).__module__}.{type(module).__qualname__}",
                "parameters": {
                    name: {"shape": list(p.shape), "dtype": str(p.dtype), "device": str(p.device)}
                    for name, p in module.named_parameters(recurse=False)
                },
                "eligible": reason is None,
                "reason": reason,
            }
        )
    return entries


def _make_replacement(module):
    """Construct a TE leaf, adopting the original native Parameter objects."""
    if module.weight.device.type != "cuda":
        raise ValueError("Move selected parameters to CUDA before TE replacement")
    import transformer_engine.pytorch as te

    if type(module) is nn.Linear:
        replacement = te.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            params_dtype=module.weight.dtype,
            device=module.weight.device,
        )
    else:
        replacement = te.LayerNorm(
            module.normalized_shape,
            eps=module.eps,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
    replacement.weight = module.weight
    if module.bias is not None:
        replacement.bias = module.bias
    replacement.train(module.training)
    return replacement


def replace_modules(model, paths):
    """Precheck a selection, prepare replacements, then update the original model.

    Returns a JSON-serializable manifest. Constructors run before module-tree
    edits, so a failed constructor leaves the original module tree intact.
    This does not establish compatibility with the surrounding model code.
    """
    paths = list(paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Select a nonempty set of unique qualified module paths")
    entries = {entry["path"]: entry for entry in inventory(model)}
    for path in paths:
        if path not in entries or not entries[path]["eligible"]:
            reason = entries.get(path, {}).get("reason", "not an inventoried leaf")
            raise ValueError(f"Cannot replace {path!r}: {reason}")
    prepared = []
    # Initialization must not advance the caller's RNG streams.
    devices = sorted(
        {
            model.get_submodule(path).weight.device.index
            for path in paths
            if model.get_submodule(path).weight.device.type == "cuda"
        }
    )
    with torch.random.fork_rng(devices=devices):
        for path in paths:
            parent_path, _, name = path.rpartition(".")
            parent = model.get_submodule(parent_path)
            prepared.append((path, parent, name, _make_replacement(model.get_submodule(path))))
    for _, parent, name, replacement in prepared:
        setattr(parent, name, replacement)
    return {
        "converted": [
            {
                **entries[path],
                "replacement": f"{type(replacement).__module__}.{type(replacement).__qualname__}",
            }
            for path, _, _, replacement in prepared
        ],
        "unconverted": [entry for path, entry in entries.items() if path not in paths],
        "quantization_enabled_by_helper": False,
    }


def main():
    """Inventory a no-argument local model factory; never convert from the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="Trusted importable module:function")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module_name, function_name = args.factory.split(":", 1)
    sys.path.insert(0, str(Path.cwd()))
    model = getattr(importlib.import_module(module_name), function_name)()
    if not isinstance(model, nn.Module):
        raise TypeError("The factory must return torch.nn.Module")
    # Exclusive creation keeps an earlier conversion record from being overwritten.
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(inventory(model), output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
