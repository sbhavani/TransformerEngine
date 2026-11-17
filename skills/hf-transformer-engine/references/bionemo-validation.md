# Biological-model conversion checks

## Test the original checkpoint

Use the original non-TE model as the reference. Loading a preconverted BioNeMo
checkpoint only checks execution of an existing port. Pin both model weights
and custom model code, retain tokenizer/processor assets, and record dependency
versions. The following opt-in tests extend `tests/pytorch/test_hf_conversion_skill.py`:

| Case | Model revision | Conversion scope |
|---|---|---|
| ESM-2 8M | `facebook/esm2_t6_8M_UR50D@c731040fcd8d73dceaa04b0a8e6329b345b0f5df` | Eligible standard projections and norms in FP32/BF16. Retain tied MLM decoder and contact head. |
| AMPLIFY 120M | `chandar-lab/AMPLIFY_120M@d918a9e8c64dc43f6ec4996022c3deba14d66470` | Attention and packed FFN projections through a SwiGLU parent adapter; preserve RMSNorm, RoPE, attention, embeddings, and decoder. |

These tests use actual protein tokenization, fixed masked-token labels, unequal
padded lengths, and two sequence-length cases. They check executed TE coverage,
hidden states, attention outputs, logits/loss, gradients, an SGD step, strict TE
state reload, native-key export, and HF checkpoint reload. ESM-2 also exercises
contact prediction. FP32/BF16 comparisons use the same dtype in both models.
BF16 checks compare both models to a native FP32 reference built from the same
BF16-rounded weights. The error budget uses twice a native-to-FP32 error
envelope plus one BF16 rounding allowance, separately for tensor RMSE and
maximum error. For AMPLIFY, the envelope covers the original separate GEMMs
and an independent PyTorch packed gate/up implementation. The latter is
checked against separate FP64 GEMMs for outputs and all gradients, so its
rounding behavior can characterize packing without depending on TE results.
FP32 AMPLIFY checks use the independently measured packed-versus-separate
PyTorch error plus FP32 rounding. Full-model FP64 attention is not claimed.
Gradients are checked per parameter. The candidate never sets its own budget.
CPU fault-injection cases require this policy to reject reversed gate order,
zero gradients, isolated corruption, and nonzero output against an exact zero
reference. This policy is specific to the evaluated fixtures, not a universal
model conversion tolerance.

ESM-2 BF16 norm replacement and AMPLIFY packed gate/up replacement can differ
from native BF16 arithmetic without changing their mathematical contracts.
Evaluate them against the FP32 reference before restricting conversion scope.
Use eager attention for ESM-2's reference, reconstruction, and HF reload so
attention outputs and backend semantics remain comparable. Disable TF32 before
starting Python for FP32 checks, since Torch and TE can otherwise select
different GEMM math modes.
Compare padded and unpadded inputs against their respective native results:
this checkpoint itself changes slightly with padding. Record that response
rather than requiring invariance absent from the reference.

The default suite also contains small ESM, Geneformer-style post-norm BERT,
Qwen3, and Mixtral configurations. Those are architecture mechanics checks,
not pretrained Geneformer/Qwen/Mixtral checkpoint validation. The already-TE
test checks no-op recognition; it does not validate a CodonFM checkpoint.

## AMPLIFY's functional projections

The pinned original AMPLIFY implementation requires xFormers and a compatible
Transformers release. Inspect its custom source before explicitly enabling
remote-code loading. It expects additive `0/-inf` masks and returns logits
without computing a labeled loss; the test computes masked cross entropy.

xFormers SwiGLU holds standard `nn.Linear` children but reads their parameters
through functional operators. Replacing those children alone can leave TE
forwards unexecuted and change the `bias=None` interface. The evaluated
[`convert_amplify.py`](../scripts/convert_amplify.py) replaces the SwiGLU parent
with explicit TE packed gate/up and down-projection calls, preserving weight
ordering and Parameter identities. Compare outputs and gradients against a
higher-precision reference to assess the different rounding from packed GEMMs.
The adapter targets the pinned 120M configuration; validate a new adapter for
other architectures or revisions.

Require all 144 declared AMPLIFY TE projection paths to execute, including both
FFN projections in every block. For ESM-2, require every block's query, key,
value, attention output, and two MLP projections. A converted classification
head alone is insufficient evidence of a model conversion.

## Run the opt-in suite

Use a GPU environment with compatible PyTorch, TE, Transformers, xFormers, and
tokenizer dependencies. The pinned AMPLIFY model uses Transformers 4 APIs;
resolve its dependency contract before loading it. Do not overwrite a working
training stack to satisfy the test: use an isolated environment.

After inspecting the pinned custom code, run from the TE repository root:

```bash
NVIDIA_TF32_OVERRIDE=0 \
NVTE_TEST_HF_BIO_PRETRAINED=1 NVTE_TEST_AMPLIFY_REMOTE_CODE=1 \
NVTE_TEST_HF_BIO_RESULTS=/tmp/te-bio-results \
python -m pytest -v tests/pytorch/test_hf_conversion_skill.py
```

The pretrained cases are disabled by default to keep routine L0 tests offline.
When explicitly enabled, missing CUDA, dependencies, or checkpoints are failures,
not successful skips. Use the frontend QA launcher's applicable backend settings.
Retain per-case JSON evidence and JUnit output. Report unsupported conversions
and unavailable tests explicitly. Benchmarks and FP8/FP4 sweeps are outside
these conversion checks.

## Reference contracts

Use BioNeMo's [model development guide](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/19ee43214fec23e87916efd468eaa62df47dba3e/models/README.md),
[ESM-2 tests](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/19ee43214fec23e87916efd468eaa62df47dba3e/models/esm2/tests/test_modeling_esm_te.py),
and [AMPLIFY tests](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/19ee43214fec23e87916efd468eaa62df47dba3e/models/amplify/tests/test_amplify_model.py)
as supporting evidence. Preserve contracts required by the user's model even
when a downstream test omits a head or an interface.
