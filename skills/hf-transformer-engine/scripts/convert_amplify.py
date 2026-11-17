# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Model-specific native-precision TE conversion for pinned AMPLIFY_120M.

Target: chandar-lab/AMPLIFY_120M@d918a9e8c64dc43f6ec4996022c3deba14d66470.
This adapter leaves attention/RoPE/mask and original RMSNorm arithmetic intact.
The checkpoint/source loader remains responsible for selecting that exact ref.
"""

from collections import Counter
import torch
from torch import nn
from torch.nn import functional as F

# Exact types exclude custom projection/parameter semantics.
# pylint: disable=unidiomatic-typecheck

REVISION = "d918a9e8c64dc43f6ec4996022c3deba14d66470"


class PackedSwiGLU(nn.Module):
    """Explicitly dispatch packed gate/up and down projections through TE."""

    def __init__(self, w12, w3):
        super().__init__()
        self.w12 = w12
        self.w3 = w3

    def forward(self, x):
        """Apply [gate; up] projection, SiLU gating, and the down projection."""
        gate, up = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * up)


def _check_projection(module):
    if type(module) is not nn.Linear:
        raise TypeError("Expected an exact native nn.Linear projection")
    if list(module.children()) or list(module.buffers()):
        raise ValueError("Projection has extra children or buffers")
    if "forward" in vars(module) or any(
        value for key, value in vars(module).items() if key.endswith("_hooks")
    ):
        raise ValueError("Projection forward/hooks require separate handling")
    if tuple(module.weight.shape) != (module.out_features, module.in_features):
        raise ValueError("Inconsistent projection dimensions")
    for parameter in module.parameters(recurse=False):
        if type(parameter) is not nn.Parameter or not parameter.is_contiguous():
            raise ValueError("Only contiguous native Parameters are supported")
        if parameter.device.type != "cuda":
            raise ValueError("Materialize the requested checkpoint on CUDA first")
        if parameter.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Native floating-point weights are required")


def _te_projection(module, te):
    converted = te.Linear(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        params_dtype=module.weight.dtype,
        device=module.weight.device,
        return_bias=False,
    )
    converted.weight = module.weight
    if module.bias is not None:
        converted.bias = module.bias
    converted.train(module.training)
    return converted


def convert_amplify(model):
    """Mutate one unwrapped pinned native AMPLIFY instance and return its manifest.

    Every original Parameter is retained, including untouched model components.
    Precheck and construct replacements before publishing module-tree edits.
    """
    import transformer_engine.pytorch as te

    if model.config.model_type != "AMPLIFY" or model.config.hidden_act.lower() != "swiglu":
        raise ValueError("This adapter targets the pinned AMPLIFY SwiGLU architecture")
    fingerprint = (
        model.config.hidden_size,
        model.config.num_hidden_layers,
        model.config.num_attention_heads,
        model.config.intermediate_size,
        model.config.vocab_size,
    )
    if fingerprint != (640, 24, 10, 2560, 27):
        raise ValueError("This evaluated adapter targets the original AMPLIFY_120M configuration")
    revision = getattr(model.config, "_commit_hash", None)
    if revision is not None and revision != REVISION:
        raise ValueError("The checkpoint revision differs from the evaluated AMPLIFY revision")
    if any(getattr(model, name, None) for name in ("hf_device_map", "peft_config", "is_quantized")):
        raise ValueError("Convert before dispatch, PEFT insertion, or quantization")
    if any(hasattr(module, "_hf_hook") for module in model.modules()):
        raise ValueError("Dispatched model requires construction-time integration")

    original_parameters = dict(model.named_parameters())
    all_parameters = list(model.named_parameters(remove_duplicate=False))
    counts = Counter(id(parameter) for _, parameter in all_parameters)
    native_keys = list(model.state_dict())
    planned = []
    for index, block in enumerate(model.transformer_encoder):
        prefix = f"transformer_encoder.{index}"
        for name in ("q", "k", "v", "wo"):
            planned.append((block, name, getattr(block, name), f"{prefix}.{name}"))
        ffn = block.ffn
        if ffn.__class__.__name__ != "SwiGLU" or getattr(ffn, "w12", None) is None:
            raise ValueError("Only original packed xformers SwiGLU is supported")
        if (
            list(ffn.buffers())
            or set(dict(ffn.named_children())) != {"w12", "w3"}
            or any(value for key, value in vars(ffn).items() if key.endswith("_hooks"))
        ):
            raise ValueError("Custom SwiGLU state/hooks require a separate adapter")
        if ffn.w12.out_features != 2 * ffn.w3.in_features:
            raise ValueError("Packed gate/up dimensions do not match down projection")
        planned.append((ffn, "w12", ffn.w12, f"{prefix}.ffn.w12"))
        planned.append((ffn, "w3", ffn.w3, f"{prefix}.ffn.w3"))
    for _, _, source, _ in planned:
        _check_projection(source)
        if any(counts[id(parameter)] != 1 for parameter in source.parameters()):
            raise ValueError("Selected projection has shared Parameters")

    devices = sorted({source.weight.device.index for _, _, source, _ in planned})
    prepared = []
    with torch.random.fork_rng(devices=devices):
        for parent, name, source, path in planned:
            prepared.append((parent, name, _te_projection(source, te), path))
    # The original xformers forward bypasses child module.forward calls.
    lookup = {path: replacement for _, _, replacement, path in prepared}
    for index, block in enumerate(model.transformer_encoder):
        prefix = f"transformer_encoder.{index}"
        for name in ("q", "k", "v", "wo"):
            setattr(block, name, lookup[f"{prefix}.{name}"])
        original_ffn = block.ffn
        block.ffn = PackedSwiGLU(lookup[f"{prefix}.ffn.w12"], lookup[f"{prefix}.ffn.w3"])
        block.ffn.train(original_ffn.training)
    current_parameters = dict(model.named_parameters())
    if set(original_parameters) != set(current_parameters) or any(
        original_parameters[name] is not current_parameters[name] for name in original_parameters
    ):
        raise RuntimeError("Conversion failed to preserve original Parameter identities")
    return {
        "adapter": "amplify-packed-swiglu-v3",
        "model_revision": REVISION,
        "converted": [path for _, _, _, path in prepared],
        "native_state_keys": native_keys,
        "unchanged": [
            "encoder",
            "decoder",
            "all RMSNorm modules",
            "attention/RoPE/mask implementation",
        ],
        "mapping": (
            "Original parameter keys and identities are retained; packed w12 first half is SiLU"
            " gate, second half is multiplicand."
        ),
        "validation": "GPU numerical validation must be performed independently",
    }


def export_native_state(model, manifest):
    """Recover an exact native-key state dict; explicitly account for TE metadata."""
    state = model.state_dict()
    expected = set(manifest["native_state_keys"])
    missing = expected - set(state)
    unexpected = set(state) - expected
    if missing or any(not name.endswith("_extra_state") for name in unexpected):
        raise ValueError(f"Unaccounted checkpoint state: missing={missing}, extra={unexpected}")
    return {name: state[name] for name in manifest["native_state_keys"]}


def save_converted(model, manifest, path):
    """Save TE state and the reconstruction manifest before HF-format export."""
    torch.save({"manifest": manifest, "state_dict": model.state_dict()}, path)


def load_converted(native_model_factory, path):
    """Recreate native architecture, apply the same adapter, and strictly load TE state."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["manifest"]["adapter"] != "amplify-packed-swiglu-v3"
        or payload["manifest"]["model_revision"] != REVISION
    ):
        raise ValueError("Checkpoint adapter version or model revision differs")
    model = native_model_factory()
    manifest = convert_amplify(model)
    if manifest["converted"] != payload["manifest"]["converted"]:
        raise ValueError("Checkpoint conversion scope differs from reconstruction")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model
