# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Build native-precision TE fusions and convert reviewed dense Qwen MLP boundaries.

Builders return a new module and an explicit relative parameter mapping. They do
not infer data flow or edit the source modules. The Qwen adapter owns the reviewed
post-attention norm -> SwiGLU boundary and preserves the HF decoder forward.
"""

from collections import defaultdict
import copy
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn

# Exact classes exclude custom forward and parameter semantics.
# pylint: disable=unidiomatic-typecheck

_SPEC = importlib.util.spec_from_file_location(
    "fusion_leaves", Path(__file__).with_name("convert_model.py")
)
_leaves = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_leaves)
ADAPTER = "qwen-post-attention-norm-mlp-v1"


def _normalization(norm):
    if not isinstance(getattr(norm, "weight", None), nn.Parameter):
        raise ValueError("Normalization requires an affine weight Parameter")
    if type(norm) is nn.LayerNorm:
        if len(norm.normalized_shape) != 1 or norm.bias is None:
            raise ValueError("LayerNorm requires one axis and both affine parameters")
        kind, eps = "LayerNorm", norm.eps
        if tuple(norm.weight.shape) != tuple(norm.normalized_shape) or tuple(
            norm.bias.shape
        ) != tuple(norm.weight.shape):
            raise ValueError("Normalization parameters do not match the normalized shape")
    elif type(norm) is getattr(nn, "RMSNorm", None):
        if len(norm.normalized_shape) != 1:
            raise ValueError("RMSNorm requires one axis")
        kind, eps = "RMSNorm", norm.eps
        if tuple(norm.weight.shape) != tuple(norm.normalized_shape):
            raise ValueError("Normalization parameters do not match the normalized shape")
        if eps is None:
            raise ValueError(
                "RMSNorm with dtype-dependent epsilon needs an explicit input-dtype adapter"
            )
    elif (type(norm).__module__, type(norm).__name__) in {
        (f"transformers.models.{family}.modeling_{family}", f"{name}RMSNorm")
        for family, name in (("qwen2", "Qwen2"), ("qwen3", "Qwen3"))
    }:
        kind, eps = "RMSNorm", norm.variance_epsilon
    else:
        raise ValueError("Unknown normalization requires a source-aware adapter")
    if not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Normalization epsilon must be finite and positive")
    if set(dict(norm.named_parameters())) != (
        {"weight", "bias"} if kind == "LayerNorm" else {"weight"}
    ):
        raise ValueError("Unexpected normalization parameters")
    return kind, eps


def _check_parts(parts):
    """Validate local contracts; callers must also exclude sharing outside this group."""
    norm = parts["norm"]
    kind, eps = _normalization(norm)
    group = nn.ModuleDict(parts)
    entries = {entry["path"]: entry for entry in _leaves.inventory(group)}
    for name in parts:
        if name != "norm" and not entries[name]["eligible"]:
            raise ValueError(f"Cannot fuse {name}: {entries[name]['reason']}")
    if _leaves._shared_parameters(group):  # pylint: disable=protected-access
        raise ValueError("Shared parameters require explicit fusion tie handling")
    if len({module.training for module in parts.values()}) != 1:
        raise ValueError("Fusion requires consistent module train/eval modes")
    for module in parts.values():
        if (
            "forward" in vars(module)
            or any(value for key, value in vars(module).items() if key.endswith("_hooks"))
            or hasattr(module, "_hf_hook")
            or list(module.buffers())
            or list(module.children())
        ):
            raise ValueError("Custom forward, hooks, buffers, or children require an adapter")
    parameters = list(group.parameters())
    if not parameters or len({(p.device, p.dtype) for p in parameters}) != 1:
        raise ValueError("Fusion requires uniform materialized parameter dtype/device")
    for parameter in parameters:
        if (
            type(parameter) is not nn.Parameter
            or parameter.is_meta
            or not parameter.is_contiguous()
        ):
            raise ValueError("Fusion requires contiguous native Parameters")
        if getattr(parameter, "_backward_hooks", None) or getattr(
            parameter, "_post_accumulate_grad_hooks", None
        ):
            raise ValueError("Parameter hooks require explicit fusion handling")
    if parameters[0].dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Fusion requires native floating-point parameters")
    if parameters[0].device.type != "cuda":
        raise ValueError("Materialize fusion parameters on CUDA first")
    if norm.weight.ndim != 1:
        raise ValueError("Normalization weight must be a vector")
    return kind, eps, group


def _initialize(parts, rows, constructor):
    """Construct without advancing RNG, copy mapped values, and preserve frozen state."""
    kind, eps, group = _check_parts(parts)
    weight = parts["norm"].weight
    with torch.random.fork_rng(devices=[weight.device.index]):
        module = constructor(kind, eps, weight)
    source = dict(group.named_parameters())
    destination = dict(module.named_parameters())
    if any(
        type(parameter) is not nn.Parameter or parameter.dtype != weight.dtype
        for parameter in destination.values()
    ):
        raise ValueError("Build native-precision fused modules outside quantized_model_init")
    if set(destination) != {target for _, target in rows}:
        raise ValueError("TE parameters differ from the supported native fusion contract")
    mapping = []
    with torch.no_grad():
        for names, target in rows:
            values = [source[name] for name in names]
            if len({value.requires_grad for value in values}) != 1:
                raise ValueError(
                    "Packed parameters must share requires_grad; rebuild optimizer after fusion"
                )
            value = torch.cat(values, dim=0) if len(values) > 1 else values[0]
            if destination[target].shape != value.shape:
                raise ValueError("TE parameter shape differs from the fusion mapping")
            destination[target].copy_(value)
            destination[target].requires_grad_(values[0].requires_grad)
            mapping.append(
                {
                    "sources": names,
                    "target": target,
                    "shapes": [list(value.shape) for value in values],
                    "dtype": str(value.dtype),
                    "requires_grad": values[0].requires_grad,
                    "transform": "concat_dim0" if len(values) > 1 else "copy",
                }
            )
    module.train(parts["norm"].training)
    return module, mapping


def build_norm_linear(norm, linear):
    """Fuse a reviewed norm -> Linear data-flow boundary into te.LayerNormLinear.

    Returns (module, relative_mapping). Source modules remain unchanged. The
    caller must check external parameter sharing and replace the entire boundary.
    """
    kind, _ = _normalization(norm)
    if type(linear) is not nn.Linear or linear.in_features != norm.weight.numel():
        raise ValueError("Linear input width must match the normalization width")
    rows = [(["norm.weight"], "layer_norm_weight"), (["linear.weight"], "weight")]
    if kind == "LayerNorm":
        rows.append((["norm.bias"], "layer_norm_bias"))
    if linear.bias is not None:
        rows.append((["linear.bias"], "bias"))

    def constructor(normalization, eps, weight):
        import transformer_engine.pytorch as te

        return te.LayerNormLinear(
            linear.in_features,
            linear.out_features,
            eps=eps,
            normalization=normalization,
            bias=linear.bias is not None,
            params_dtype=weight.dtype,
            device=weight.device,
            return_bias=False,
            return_layernorm_output=False,
        )

    return _initialize({"norm": norm, "linear": linear}, rows, constructor)


def build_norm_mlp(norm, gate, up, down):
    """Fuse norm -> SiLU(gate) * up -> down into te.LayerNormMLP (SwiGLU).

    FC1 rows are [gate; up], with no interleaving. Source Parameters are copied
    into new packed Parameters, so optimizer construction must follow fusion.
    """
    kind, _ = _normalization(norm)
    if any(type(module) is not nn.Linear for module in (gate, up, down)):
        raise ValueError("MLP fusion requires exact native Linear projections")
    hidden, intermediate = norm.weight.numel(), gate.out_features
    if (gate.in_features, up.in_features, up.out_features, down.in_features, down.out_features) != (
        hidden,
        hidden,
        intermediate,
        intermediate,
        hidden,
    ):
        raise ValueError("SwiGLU projection dimensions do not match")
    biases = [module.bias is not None for module in (gate, up, down)]
    if len(set(biases)) != 1:
        raise ValueError("TE MLP requires consistent projection bias presence")
    rows = [
        (["norm.weight"], "layer_norm_weight"),
        (["gate.weight", "up.weight"], "fc1_weight"),
        (["down.weight"], "fc2_weight"),
    ]
    if kind == "LayerNorm":
        rows.append((["norm.bias"], "layer_norm_bias"))
    if biases[0]:
        rows.extend([(["gate.bias", "up.bias"], "fc1_bias"), (["down.bias"], "fc2_bias")])

    def constructor(normalization, eps, weight):
        import transformer_engine.pytorch as te

        return te.LayerNormMLP(
            hidden,
            intermediate,
            eps=eps,
            normalization=normalization,
            activation="swiglu",
            bias=biases[0],
            params_dtype=weight.dtype,
            device=weight.device,
            return_bias=False,
            return_layernorm_output=False,
        )

    return _initialize({"norm": norm, "gate": gate, "up": up, "down": down}, rows, constructor)


def convert_qwen_mlp(model):
    """Fuse post-attention RMSNorm/SwiGLU in official dense HF Qwen2/Qwen3 decoders.

    Inspect the installed HF decoder source before use. Attention, residuals,
    RoPE, masks, and cache orchestration remain in the original decoder forward.
    All replacements are prepared before tree mutation. Optimizer/PEFT/dispatch
    integration must happen afterward; unsupported contracts fail explicitly.
    """
    family = model.config.model_type
    if family not in ("qwen2", "qwen3") or model.config.hidden_act != "silu":
        raise ValueError("This adapter targets dense Qwen2/Qwen3 with SiLU SwiGLU")
    if any(
        getattr(model, flag, None)
        for flag in (
            "hf_device_map",
            "is_quantized",
            "is_loaded_in_4bit",
            "is_loaded_in_8bit",
            "peft_config",
        )
    ):
        raise ValueError("Fuse before dispatch, quantization, or PEFT insertion")
    for module in model.modules():
        if (
            hasattr(module, "_hf_hook")
            or "forward" in vars(module)
            or any(value for key, value in vars(module).items() if key.endswith("_hooks"))
        ):
            raise ValueError("Model hooks/custom forwards require an explicit fusion adapter")
    shared = _leaves._shared_parameters(model)  # pylint: disable=protected-access
    namespace = f"transformers.models.{family}.modeling_{family}"
    name = "Qwen2" if family == "qwen2" else "Qwen3"
    blocks = [
        (path, block)
        for path, block in model.named_modules()
        if type(block).__module__ == namespace and type(block).__name__ == f"{name}DecoderLayer"
    ]
    if not blocks or len(blocks) != model.config.num_hidden_layers:
        raise ValueError("Expected one unwrapped native decoder per configured Qwen layer")
    native_keys = list(model.state_dict())
    native_buffers = _buffer_contract(model)
    model_config = _config_contract(model)
    native_requires_grad = {
        key: parameter.requires_grad
        for key, parameter in model.named_parameters(remove_duplicate=False)
    }
    native_ties = _parameter_ties(model)
    prepared, mappings, specifications = [], [], []
    for path, block in blocks:
        norm, mlp = block.post_attention_layernorm, block.mlp
        if (type(norm).__module__, type(norm).__name__) != (namespace, f"{name}RMSNorm"):
            raise ValueError("Expected the native Qwen RMSNorm at the reviewed MLP boundary")
        if (type(mlp).__module__, type(mlp).__name__) != (namespace, f"{name}MLP"):
            raise ValueError(
                "Expected original native Qwen MLP; do not fuse an already converted model"
            )
        if set(dict(mlp.named_children())) != {"gate_proj", "up_proj", "down_proj", "act_fn"}:
            raise ValueError("Custom MLP children require explicit fusion handling")
        if (
            list(mlp.buffers())
            or list(mlp.named_parameters(recurse=False))
            or norm.training != mlp.training
        ):
            raise ValueError("Custom MLP state or inconsistent train/eval modes require an adapter")
        if (type(mlp.act_fn).__module__, type(mlp.act_fn).__name__) not in (
            ("transformers.activations", "SiLUActivation"),
            ("torch.nn.modules.activation", "SiLU"),
        ):
            raise ValueError("Expected the native SiLU activation")
        prefixes = {
            "norm": f"{path}.post_attention_layernorm",
            "gate": f"{path}.mlp.gate_proj",
            "up": f"{path}.mlp.up_proj",
            "down": f"{path}.mlp.down_proj",
        }
        if any(
            key.startswith(tuple(prefix + "." for prefix in prefixes.values())) for key in shared
        ):
            raise ValueError(
                "Selected parameters share identity/storage outside the fused boundary"
            )
        fused, rows = build_norm_mlp(norm, mlp.gate_proj, mlp.up_proj, mlp.down_proj)
        for row in rows:
            row["sources"] = [
                prefixes[key.split(".", 1)[0]] + "." + key.split(".", 1)[1]
                for key in row["sources"]
            ]
            row["target"] = f"{path}.mlp.{row['target']}"
        mappings.extend(rows)
        specifications.append(
            {
                "path": path,
                "normalization": "RMSNorm",
                "eps": norm.variance_epsilon,
                "hidden_size": mlp.gate_proj.in_features,
                "ffn_size": mlp.gate_proj.out_features,
            }
        )
        prepared.append((block, fused, norm.training))
    for block, fused, training in prepared:
        block.post_attention_layernorm = nn.Identity().train(training)
        block.mlp = fused
    return {
        "adapter": ADAPTER,
        "layers": specifications,
        "mapping": mappings,
        "native_state_keys": native_keys,
        "native_buffers": native_buffers,
        "model_config": model_config,
        "attention_implementation": model.config._attn_implementation,
        "native_requires_grad": native_requires_grad,
        "native_ties": native_ties,
        "source_class": f"{type(model).__module__}.{type(model).__name__}",
        "source_model_type": family,
        "source_model_revision": getattr(model.config, "_commit_hash", None),
        "converted": [f"{path}.mlp" for path, _ in blocks],
        "rebuild_optimizer": True,
        "quantization_enabled_by_helper": False,
    }


def export_native_state(model, manifest):
    """Invert fusion packing and renaming, rejecting unaccounted checkpoint state."""
    state = model.state_dict()
    mapped = {name for row in manifest["mapping"] for name in row["sources"]}
    unchanged = set(manifest["native_state_keys"]) - mapped
    expected = unchanged | {row["target"] for row in manifest["mapping"]}
    missing, extra = expected - set(state), set(state) - expected
    if missing or any(not key.endswith("_extra_state") for key in extra):
        raise ValueError(f"Unaccounted fusion state: missing={missing}, extra={extra}")
    result = {key: state[key] for key in unchanged}
    for row in manifest["mapping"]:
        value = state[row["target"]]
        values = (
            value.split([shape[0] for shape in row["shapes"]], dim=0)
            if len(row["sources"]) > 1
            else [value]
        )
        for source, part, shape in zip(row["sources"], values, row["shapes"]):
            if source in result or list(part.shape) != shape:
                raise ValueError("Fusion mapping collision or tensor shape mismatch")
            result[source] = part
    if set(result) != set(manifest["native_state_keys"]):
        raise ValueError("Fusion export did not cover all original keys")
    return result


def _parameter_ties(model):
    groups = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups[id(parameter)].append(name)
    return sorted(sorted(names) for names in groups.values() if len(names) > 1)


def _buffer_contract(model):
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in model.named_buffers(remove_duplicate=False)
    }


def _config_contract(model):
    return {
        name: value
        for name, value in model.config.to_dict().items()
        if not name.startswith("_") and name not in ("transformers_version", "torch_dtype", "dtype")
    }


def save_converted(model, manifest, path):
    """Save TE state together with its required reconstruction mapping."""
    saved = copy.deepcopy(manifest)
    targets = {source: row["target"] for row in saved["mapping"] for source in row["sources"]}
    saved["native_requires_grad"] = {
        name: model.get_parameter(targets.get(name, name)).requires_grad
        for name in saved["native_requires_grad"]
    }
    for row in saved["mapping"]:
        row["requires_grad"] = model.get_parameter(row["target"]).requires_grad
        row["dtype"] = str(model.get_parameter(row["target"]).dtype)
    buffers = _buffer_contract(model)
    saved["native_buffers"] = {name: buffers[name] for name in saved["native_buffers"]}
    torch.save({"manifest": saved, "state_dict": model.state_dict()}, path)


def load_converted(native_model_factory, path):
    """Reconstruct reviewed Qwen fusions and strictly load their checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["manifest"]["adapter"] != ADAPTER:
        raise ValueError("Fusion adapter version differs")
    model = native_model_factory()
    if _buffer_contract(model) != payload["manifest"]["native_buffers"]:
        raise ValueError("Native factory buffer shapes/dtypes differ; preserve FP32 RoPE buffers")
    if (
        _config_contract(model) != payload["manifest"]["model_config"]
        or model.config._attn_implementation != payload["manifest"]["attention_implementation"]
    ):
        raise ValueError("Native factory model configuration or attention backend differs")
    parameters = dict(model.named_parameters(remove_duplicate=False))
    if (
        set(parameters) != set(payload["manifest"]["native_requires_grad"])
        or _parameter_ties(model) != payload["manifest"]["native_ties"]
    ):
        raise ValueError("Native factory parameter names or ties differ from checkpoint")
    for name, requires_grad in payload["manifest"]["native_requires_grad"].items():
        parameters[name].requires_grad_(requires_grad)
    manifest = convert_qwen_mlp(model)
    if any(
        manifest[key] != payload["manifest"][key]
        for key in (
            "source_class",
            "source_model_type",
            "native_requires_grad",
            "native_ties",
            "native_buffers",
            "model_config",
            "attention_implementation",
            "layers",
            "mapping",
            "native_state_keys",
        )
    ):
        raise ValueError("Fusion reconstruction contract differs from checkpoint")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model
