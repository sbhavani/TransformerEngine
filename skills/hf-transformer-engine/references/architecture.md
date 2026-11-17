# Match computation and interfaces

Inspect the installed HF modeling source, not just `config.model_type`. In
dynamic models, combine source inspection with a run over representative inputs
and module hooks that are removed after profiling.

## Architecture contract

Record these before choosing a fused module:

- Norm type, epsilon, affine/bias policy, accumulation dtype, zero-centered
  gamma, pre/post-norm placement, and any parallel residual branches.
- Projection weight orientation and biases; activation function and exact
  variant (including approximate GELU); gated MLP multiplication and ordering.
- Query/key/value dimensions, head dimension, GQA/MQA groups, projection layout,
  QK norms, attention scaling/soft-capping, and output projection.
- Position encoding, RoPE rotation convention, theta, scaling, offsets, and
  dynamic sequence-length handling; relative position bias or ALiBi if present.
- Causal, bidirectional, sliding-window, padding, cross-attention, or packed
  masks; mask polarity and representation at the actual TE call boundary.
- Input/output layout and dtype, dropout placement/probabilities, residual
  scaling, output container type, and forward keyword arguments.
- KV-cache type/update semantics, `cache_position`, `use_cache`,
  `output_attentions`, hidden-state collection, and generation entry points.
- Weight sharing, hooks, LoRA/PEFT, quantized weights, offload/device maps,
  tensor parallelism, and compile/checkpointing wrappers.

Matching hidden dimensions alone establishes none of these contracts. A
`**kwargs` wrapper must implement or explicitly reject meaningful unsupported
arguments; silently discarding cache/mask/position arguments is incorrect.

## Choose the replacement scope

| Model structure | Candidate path | Contracts that need particular attention |
|---|---|---|
| Dense decoder | Full block or norm/MLP/attention fusion | GQA, RoPE convention/scaling, residual ordering, cache and generation |
| Encoder, masked LM, vision encoder | Full block where equivalent; otherwise assembled attention/MLP/norm | Pre/post-norm, noncausal masks, activation variant, pooling/head outputs |
| Encoder–decoder | Separate encoder and decoder adapters, or partial fusion | Cross-attention, relative biases, independent masks, cache layout |
| MoE | Attention/norm/projections first; expert fusion when equivalent | Router numerics, top-k/order, shared experts, auxiliary losses, communication |
| Multimodal/hybrid/state-space/custom | Compatible operations around custom core | Modality-specific positions/masks and non-attention state transitions |

Check current TE capabilities before ruling out a path. TE supports more than a
single generic attention block, but a capability's presence does not prove it
implements the target model's exact variant. Keep irreducible custom operations
and their surrounding interfaces intact.

Prefer focused TE modules when an entire block differs. For example, a norm
after a residual addition is not interchangeable with a norm before attention.
Use the target-version `TransformerLayer` configuration only after checking its
ordering; otherwise compose `Linear`, norms, attention, and MLP operations.

## Reference implementations

- [TE Llama tutorial](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_llama/tutorial_accelerate_hf_llama_with_te.ipynb)
  and its sibling `te_llama.py`: architecture-specific block wrapper and
  pretrained state mapping. Read the target ref's executable source as well as
  notebook prose. Check generation interfaces separately from its fine-tuning example.
- [BioNeMo acceleration skill](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/fbae1941d8cba50105c8379c810d1c95396eb26c/skills/bionemo-recipes-acceleration/SKILL.md):
  reference for choosing conversion depth and validating a port. Its recipe
  repository layout, CI harness, version pins, and hardware restrictions do not
  become TE requirements. Consult relevant model/converter examples on demand.
- [Accelerate TE integration](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/transformer_engine.py):
  an existing automatic operation conversion route. Inspect the installed
  version before use, particularly filtering, parameter copying, shared weights,
  and how its conversion interacts with a custom port.

Resolve APIs from the installed versions or matching source refs. Do not copy
the Llama tutorial's internal Transformers imports or class monkey-patching into
an unrelated model; prefer a local adapter/construction path with explicit scope.
