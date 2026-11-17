# Generate a fused conversion for the requested architecture

Start from the model's computation and the installed TE API. Generate a local
adapter for the model being converted. The examples below illustrate reusable
patterns; a missing architecture name or a bundled helper's rejection does not
establish that TE cannot accelerate that model.

## Reason about fusion candidates

Read the installed model's forward implementation. Module names and adjacent
positions in `named_modules()` do not establish data flow. The norm must feed
the projection or complete MLP being fused, with the same residual placement,
epsilon, activation, and affine semantics. Inspect other consumers and hooks.

Use these questions to develop the conversion:

1. What computation does this subgraph perform, including normalization, residual
   placement, activation, biases, branches, layout, and returned values?
2. Which installed TE module can express that computation? Consider
   `LayerNormLinear`, `LayerNormMLP`, focused attention modules, or
   `TransformerLayer`, including their supported normalization/activation options.
3. What wrapper preserves the model's surrounding forward contract? Account for
   every consumer of an intermediate before removing or moving its producer.
4. How do the original parameters map to the selected TE configuration, and how
   will reconstruction and inverse export work?
5. Which focused numerical and interface tests would falsify the proposed mapping
   or fusion boundary? Implement and run them against the original model.

Produce runnable code in the user's project from that analysis. Reuse existing
integration code where appropriate. If a bundled helper does not support a
valid pattern, generate a custom adapter using the TE API; retain the helper's
checks for its own supported inputs. A new architecture should not require
adding its name to this skill or weakening an existing helper's checks.

If a computation has no compatible TE expression, preserve that portion and
explain the concrete mismatch. Do not force fusion by changing the model's math.

## Optional reusable builders

`scripts/fused.py` implements two reviewed patterns:

| Native computation | Builder | TE module |
|---|---|---|
| LayerNorm/RMSNorm → one Linear | `build_norm_linear(norm, linear)` | `LayerNormLinear` |
| LayerNorm/RMSNorm → SiLU(gate) × up → down | `build_norm_mlp(norm, gate, up, down)` | `LayerNormMLP(activation="swiglu")` |

Import it with `importlib.util.spec_from_file_location`, as with `convert_model.py`.
Keep `convert_model.py` beside `fused.py` if copying the helpers into a project.
Other activations, normalization definitions, or layouts may be expressible
through TE even though these particular builders do not implement them.

Each builder returns `(fused_module, relative_mapping)` and leaves the source
modules unchanged. The mappings name all source and target parameters and
record shapes/dtype. FC1 uses `[gate; up]` along dimension zero. Review and
qualify those mapping paths when installing the module in a model wrapper.
Replace the complete reviewed boundary so normalization runs exactly once.

For example, after checking that a parent computes `self.linear(self.norm(x))`:

```python
fused, mapping = fusion.build_norm_linear(parent.norm, parent.linear)
# Persist this construction in the parent's implementation:
parent.norm = torch.nn.Identity().train(parent.norm.training)
parent.linear = fused
# Qualify mapping source/target paths for this parent and record the native keys.
```

These builders do not infer a whole model graph. They accept standard affine
LayerNorm, PyTorch RMSNorm, and the reviewed HF Qwen RMSNorm classes. Parameters
must be materialized on CUDA with consistent native dtype/device. Custom
normalization, additional state/hooks, sharing, or unsupported dimensions need
a source-aware adapter. PyTorch RMSNorm requires an explicit epsilon: its
default is input-dtype dependent and cannot be replaced by one fixed TE epsilon
without an input-dtype contract. Check ties outside the supplied group before using a
builder; the Qwen adapter performs that check against the whole model.

Fusion copies values into new Parameters and changes parameter names. Build
the optimizer after conversion. Frozen state is preserved; gate/up parameters
packed into one Parameter must have matching `requires_grad`. Optimizer-state
migration, PEFT, dispatch/offload, distributed wrapping, compilation, and
quantized parameter construction require separate integration at the right
construction boundary. Do not silently remove existing integrations.
The Qwen TE checkpoint records trainable flags and parameter ties; reconstruction
restores the saved flags and rejects a native factory with different ties.
It also checks the native configuration, attention backend, and buffer shapes/dtypes.

## Worked example: dense Qwen2 and Qwen3

The bundled adapter targets official dense HF Qwen2/Qwen3 decoder classes with
SiLU SwiGLU. It fuses each post-attention RMSNorm plus MLP into one
`LayerNormMLP(normalization="RMSNorm", activation="swiglu")`. The old post-attention
norm becomes Identity, so the original decoder forward keeps its residual
connections. Attention, input/QK norms, RoPE, masks, task heads, and KV-cache
orchestration remain native. Untouched embedding/head ties are preserved.

```python
import copy
from transformers import AutoModelForCausalLM

model = load_requested_hf_checkpoint(..., torch_dtype=torch.bfloat16).cuda()
saved_config = copy.deepcopy(model.config)
manifest = fusion.convert_qwen_mlp(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=...)
fusion.save_converted(model, manifest, "fused.pt")

# Factory preserves parameter dtype AND intentional buffer precision, e.g.:
def native_model_factory():
    return AutoModelForCausalLM.from_config(
        saved_config, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
restored = fusion.load_converted(native_model_factory, "fused.pt")

native = native_model_factory()
native.load_state_dict(fusion.export_native_state(model, manifest), strict=True)
native.save_pretrained("hf_export")
tokenizer.save_pretrained("hf_export")
```

Preserve buffers intentionally kept in FP32, especially RoPE inverse frequencies.
Loading/constructing weights with `torch_dtype=...` and then moving only the
device preserves that distinction. A blanket `model.to(dtype=...)` can downcast
nonpersistent positional buffers that are absent from `state_dict`, breaking
otherwise strict checkpoint reloads. The loader rejects a mismatched factory.

Do not combine this adapter with a prior leaf conversion of the same norm/MLP.
Prepare all fusions before mutating the tree, and reject unsupported model
state rather than applying only some layers silently. Inspect the installed HF
decoder source before using the adapter on another Transformers release.
For other dense architectures, generate a boundary adapter using these builders
and preserve their own forward and state contracts; do not relabel them Qwen.

Qwen's input norm feeds multiple attention projections. Deleting it while
fusing only one projection changes the other branches. A future QKV fusion
needs its own wrapper, head grouping/mapping, and validation. The bundled
Qwen adapter currently fuses the norm–MLP boundary.

## Validate the fused path

Use [validation.md](validation.md). Instrument the actual fused TE module to
prove execution; successful leaf conversion does not establish fusion coverage.
Compare outputs, loss, input/parameter gradients through the inverse mapping,
an optimizer step, strict TE reload, native HF export, and cached decoding.
Keep checkpoint tensors exact and calibrate floating-point comparisons.

Fusion changes rounding locations: normalization affine operations and SwiGLU
can execute in FP32 before output rounding, and packed GEMMs change backward
accumulation. The tests use a separate native PyTorch fusion formulation to
characterize those effects, checked against unfused FP64 math. BF16 model
comparisons also use the same rounded checkpoint weights promoted to FP32.
Set budgets from these independent references, never from the TE result.
The fused fixtures retain the same reference-based RMSE check. Their BF16
gradient maximum-error allowance covers both forward and backward stores:
four unit-roundoff contributions for norm–linear and eight for norm–MLP.
Norm–MLP has norm, FC1, activation, and FC2 outputs in forward, followed by
FC2 input, activation, FC1 input, and norm input gradients in backward.
A gradient depends on rounding in both passes. This is not an
architecture-independent error guarantee. A sparse-corruption test verifies
that the outlier check still catches errors that tensor RMSE would hide.
For Qwen norm-weight gradients, the FP32 reference also measures the sum of
absolute per-token gradient contributions. Cancellation can make the final
reduction much smaller than those operands, so that independent sensitivity
scale is used for the maximum-error rounding term. The tensor-wide RMSE
criterion remains unchanged; missing gradients still fail.
For large vocabularies, the test computes error statistics on CPU to avoid
keeping several full-size FP32 gradient temporaries on the validation GPU.
This changes only where statistics are evaluated, not the conversion math.

The focused suite includes both builders with LayerNorm/RMSNorm, bias on/off,
FP32/BF16, small Qwen2/Qwen3 models, and an opt-in pinned pretrained case:
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.

```bash
NVIDIA_TF32_OVERRIDE=0 NVTE_TEST_HF_QWEN_PRETRAINED=1 \
NVTE_TEST_HF_BIO_RESULTS=/tmp/te-fusion-results \
python -m pytest -v tests/pytorch/test_hf_conversion_skill.py -k 'fusion or fused'
```

The pretrained case downloads public weights and fails if explicitly enabled
without CUDA or required dependencies. It does not enable remote custom code.
Record the installed Transformers/TE versions and the adapter manifest. These
checks validate conversion mechanics, without performance or FP8/FP4 gates.
