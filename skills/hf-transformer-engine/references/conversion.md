# Implement and integrate conversion

## Inventory an existing model

The helper operates on an already constructed `torch.nn.Module`, so it works
without prescribing a particular HF task class or downloading a checkpoint.
Run it from the model project so the trusted model factory is importable:

```bash
python /absolute/path/to/skill/scripts/convert_model.py \
  --factory model_setup:load_model --output te-inventory.json
```

The factory takes no arguments and returns the existing model. It may load the
user's chosen checkpoint; the helper itself does not access the Hub or enable
remote code. Importing/calling a factory executes that project's code, so use a
known local entry point. The CLI only inventories; it does not apply conversion.

Each leaf's qualified path, type, parameter shape/dtype/device, eligibility, and
reason is recorded. Eligibility describes the helper's mechanical coverage.
Verify that the surrounding HF code accepts a replacement: inspect callers for
type checks and functional access to
weights, in addition to ordinary `module(x)` calls.

## Apply selected operation replacements

Import `scripts/convert_model.py` with `importlib.util.spec_from_file_location`
or copy this small helper into the target project with its license header. Make
the integration persistent in the model construction code, for example:

```python
model = load_original_checkpoint(..., torch_dtype=torch.bfloat16).to(device="cuda")
before = converter.inventory(model)
selected_paths = ["model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj"]
manifest = converter.replace_modules(model, selected_paths)
# Construct the optimizer and Trainer/Accelerate/distributed wrappers afterward.
```

These names are examples, not universal HF paths. Generate the selection from
the actual inventory and caller contracts; use the same deterministic selection
when reconstructing the converted architecture. An agent can select all reviewed
eligible paths programmatically, then validate their interfaces and numerical behavior.

The helper prechecks every requested path and prepares replacements before
editing the module tree. It only accepts exact standard Linear and single-axis
affine LayerNorm with both affine parameters. It preserves Parameter identity,
values, dtype/device, `requires_grad`, and module train/eval state. It rejects
shared modules/parameters/storage, nonstandard hooks/state, custom subclasses,
unsupported tensors, meta tensors, and dispatched/quantized/PEFT models; these
need a model-aware adapter. Only CUDA materialized parameters can be replaced.
The root module itself is not replaced. Review the manifest's skipped paths
and document any components that need architecture-specific handling.

Do not apply the helper inside `quantized_model_init`: its contract is adoption
of native-precision parameters. Parameter-quantized construction needs a
separate validated initialization/checkpoint path. Build the optimizer after
conversion even when identity is preserved; deeper fusion typically changes
parameter grouping and identity. Existing optimizer state requires explicit
mapping and validation rather than silent reuse.

## Fused adapters and state mapping

Read [fused-conversion.md](fused-conversion.md) for runnable `LayerNormLinear`
and SwiGLU `LayerNormMLP` builders and the dense Qwen2/Qwen3 norm–MLP adapter.
Treat these as optional reusable implementations. For the requested model,
generate the adapter from its actual data flow and the target TE API; reuse a
builder when its contract fits.

For each block or fused submodule, create a local wrapper with the original
forward interface and output structure. Explicitly translate masks, layout,
positions, and cache state at the boundary. Do not change the HF config's
architecture identity and assume AutoModel can now reconstruct that wrapper.

Write a mapping table with source key(s), destination key(s), transformation,
expected shape/dtype, and inverse transformation when export is needed. Examples:

- Separate Q/K/V to a fused tensor: derive head grouping and interleaving from
  the selected TE configuration. A naive concatenation is not universally valid.
- HF `Conv1D`: verify its transposed weight convention before mapping to Linear.
- SwiGLU gate/up projection fusion: verify the activation operand, concatenation
  order, and any interleaving in the target kernel.
- RMSNorm: verify epsilon, accumulation dtype, weight offset, and zero-centered
  gamma. A matching class name is insufficient for a general converter.

Use `torch.no_grad()` and `copy_` for explicit value transforms, not unchecked
`.data` assignment. Track every destination parameter and every intentionally
unused source key. Detect collisions and missing parameters. A partial port must
also load the unchanged modules. Check ties by identity and storage relationships
after mapping; equal tensor values alone do not preserve sharing.

## Save and reload

TE state dicts can contain extra quantization state and different parameter keys.
Persist model/config revision, selected paths or adapter version, mapping version,
and dtype alongside the converted checkpoint. Reconstruct the TE model
through the same explicit construction/conversion path before loading its state.
Exercise strict loading in that path and compare results after reload.

Do not promise ordinary HF `save_pretrained`/`from_pretrained` interoperability
without testing it. If HF-format export is required, map native parameters back
to a fresh HF instance and save through its supported API. Treat TE quantizer
metadata as TE state, not an extra HF weight. Validate restoration of optimizer
and TE extra state when training continuation is in scope.
