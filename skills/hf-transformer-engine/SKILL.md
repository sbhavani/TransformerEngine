---
name: hf-transformer-engine
description: "Adapt PyTorch Hugging Face model architectures to NVIDIA Transformer Engine (TE). Inspect model forward paths, identify compatible norm/projection, MLP, attention, and Transformer-block fusions, generate model-specific adapters and weight mappings, and validate numerics, model interfaces, and checkpoint reloads."
---

# Hugging Face Model Conversion to Transformer Engine

## When to Use This Skill

Use when asked to port or accelerate an existing Hugging Face model with TE.
Work with the requested architecture, including encoders, decoders,
encoder–decoders, multimodal models, and models with custom blocks. Select
compatible operations within each model; a universal full-block replacement
cannot be inferred from a model name or matching tensor shapes.

The result is runnable conversion code in the user's project, explicit weight
mapping, numerical checks, and a reload path. Completion means a correct
conversion in the model's existing FP32/BF16/FP16 mode. Performance measurement
and FP8/FP4 recipe evaluation are separate follow-up tasks.
TE owns optimized building blocks; retain the existing
model, task heads, data pipeline, Trainer/Accelerate integration, and training
or generation interface unless changing them is part of the request.

Drive conversion by inspecting the requested model and generating an integration
for its actual computation. The bundled scripts are optional implementation aids
and worked examples; their supported classes do not define the skill's scope.

## Workflow

### 1. Identify the model and establish a baseline

Resolve the model ID/local checkpoint and revision, Transformers and TE versions,
actual Python model class, task, parameter/buffer dtypes, input shapes, and target GPU.
Preserve intentional FP32 buffers such as RoPE frequencies when materializing
weights in the requested dtype. Inspect the
model's real forward path and config, including custom remote code if used.
Do not enable `trust_remote_code` or substitute a different checkpoint silently.
Use the user's environment; fix missing TE installation separately rather than
replacing their framework stack. Check target-version TE source/docs, not a
historical support table.

Preserve a runnable unmodified baseline with identical weights, tokenizer or
processor, inputs, masks, and loss. A tiny random model from the same config is
useful for mechanics tests but does not prove pretrained-model quality. If the
checkpoint or GPU is unavailable, still produce the adapter and reproducible
validation commands; state which checks remain unrun.

### 2. Inventory and choose the conversion depth

Read [architecture.md](references/architecture.md). Inventory the live model
using `scripts/convert_model.py` as described in
[conversion.md](references/conversion.md). The inventory does not import TE,
download weights, mutate modules, or prove execution compatibility.

Choose per submodule:

| Depth | Use when | Implementation |
|---|---|---|
| Individual operations | Standard Linear/LayerNorm contracts match | Apply explicit qualified paths with the bundled helper, or use the target stack's existing converter if its contracts suffice. |
| Fused submodules | Norm–projection, MLP, or attention semantics match | Generate a boundary adapter from the actual forward path; reuse a bundled builder or existing integration when its contract matches. |
| Full Transformer blocks | Residual, norm, attention, MLP, and forward interfaces all match | Generate a model-specific wrapper around `te.TransformerLayer` and map all parameters. |

Pursue useful fusion when the architecture supports it; do not stop at leaf
swaps merely because they are easier. If only part of a model matches, convert
that part and list the remaining components. No compatible operations is a
valid outcome; do not force a different mathematical model to claim coverage.

For fused conversions, read [fused-conversion.md](references/fused-conversion.md).
Trace candidate computations, compare their semantics with the installed TE
modules, and generate the wrapper and mapping for the requested architecture.
Use `scripts/fused.py` when its builders fit. Its Qwen adapter demonstrates one
reviewed norm–MLP boundary; extend the approach to other models by reasoning
about their source rather than requiring a recognized architecture name.

For BioNeMo-relevant models, read
[bionemo-validation.md](references/bionemo-validation.md) for pinned biological
checkpoint tests, functional weight-access pitfalls, and the evaluated AMPLIFY
operation adapter. An already-TE model needs recognition and compatibility
checks, not a second conversion.

### 3. Implement the conversion and weight mapping

Read [conversion.md](references/conversion.md). Use the existing Accelerate TE
integration when it meets the requested behavior; inspect the installed version's
conversion, ties, filtering, and checkpoint handling before relying on it. Do not
apply both its automatic conversion and a custom conversion to the same modules.

For explicit operation swaps, the bundled `inventory()` and
`replace_modules()` helpers support exact standard Linear and affine
single-axis LayerNorm modules, preserve their Parameter objects and train/eval
state, and reject cases requiring an adapter. This helper is a starting point;
custom norms, HF Conv1D, attention, adapters, and fused blocks require source-aware
code generated for the requested model.

Apply conversion after weights are materialized and before optimizer creation,
PEFT injection, distributed wrapping, dispatch/offload, or compilation. If the
user already uses these features, integrate at the correct construction boundary
and explicitly test them. Do not unwrap or discard them to make conversion pass.
Fused builders copy/pack parameters and change their names; rebuild the optimizer
and use the supplied mapping and reconstruction path rather than assuming the
identity-preserving leaf-conversion contract applies.

For deeper conversion, produce an explicit mapping for each renamed, reshaped,
transposed, concatenated, or split tensor, including inverse mapping when HF
checkpoint export is required. Account for unchanged parameters too. Detect
missing/unexpected keys and collisions; never hide unmapped weights behind an
unchecked `strict=False`. Preserve tied embeddings/heads and frozen parameters.

### 4. Validate conversion mechanics

Read [validation.md](references/validation.md). First compare the converted
model against the original in the same FP32/BF16/FP16 mode with quantization
disabled. Test individual converted blocks and end-to-end task outputs/loss,
gradients, an optimizer step, and save/reload behavior. Test padding, masks,
generation/cache behavior, and other interfaces actually used by the task.
Keep structural and checkpoint checks exact. For floating-point differences,
follow the reference-based tolerance guidance before rejecting a conversion;
native BF16 arithmetic is not an exact numerical oracle.

Confirm that the selected TE replacements execute on the tested path. Conversion
does not require a throughput benchmark or a quantization recipe sweep.

### 5. Deliver the conversion

Deliver the integration code, conversion manifest/state mapping, tests,
reload instructions, and a concise report of correctness results and limitations.
Record converted and skipped paths, dtype, exact commands,
hardware/software versions, and checks that were unavailable. If a numerical
check fails, repair or revert the affected conversion before presenting it as
usable. Never describe an unexecuted port as validated.
