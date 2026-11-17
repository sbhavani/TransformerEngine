# Validate conversion mechanics

## Native-precision checks

Keep original and converted runs independent: do not compare two references to
the same mutated object. Reload identical baseline weights in separate processes
if two full models do not fit in memory. Record seeds and math settings, and
preserve the tokenizer/processor, batch, labels, padding, and attention backend.

1. Compare converted operation/block outputs and applicable input/parameter
   gradients against native HF/PyTorch results in the same dtype. Use named
   state mapping to compare fused parameter gradients.
2. Compare end-to-end task outputs and loss. Compare logits, not just argmax or
   output shapes; for multimodal/non-language models use their actual task
   quantities. Check finite outputs/gradients and frozen parameters.
3. Compare a short training step and updated weights using equivalent optimizer
   hyperparameters. Disable dropout for deterministic parity, then separately
   test training with intended dropout settings; fused kernels may consume RNG
   differently. Gradient checkpointing needs recomputation/backward coverage.
4. Exercise representative masks, padded and unpadded inputs, variable lengths,
   and model-specific boundaries. For encoder–decoder models include real
   cross-attention inputs. For multimodal models retain modality-specific masks.
5. If generation is required, test prefill and incremental decoding, cache on/off,
   multiple prompt lengths and padding patterns, and the actual generation API.
   Verify cache positions, updated state, return types, and requested attention
   outputs. Beam search needs cache reordering coverage when used.
6. Save, reconstruct, strictly reload, and compare outputs. Check tied weights
   and state mapping round trips. If resuming training is requested, validate
   optimizer state restoration through the parameter mapping too.

### Distinguish rounding from conversion defects

Keep weight mapping, shapes, parameter ties, frozen state, and checkpoint tensor
round trips exact. Floating-point outputs and gradients generally need numerical
agreement rather than bitwise equality: fusion, reductions, and accumulation
order can change rounding even when the mathematical operation is preserved.

Start with the same dtype and math settings in both models, with quantization
disabled. If a comparison fails, isolate the operation and verify weight order,
masks, residuals, positions, and interfaces. A pointwise tolerance failure alone
does not establish that a conversion is invalid.

For reduced-precision arithmetic, use a third, independent FP32 or FP64 native
reference when practical. Promote the **same rounded weights and inputs**, so
the comparison isolates arithmetic rather than different checkpoint precision.
Measure native-to-reference and converted-to-reference error for outputs, loss,
and applicable gradients. A converted result closer to that reference should
not fail solely because it differs from the native reduced-precision result.
Align attention/math backends and disable TF32 for an IEEE FP32 reference.

Use tensor RMSE and an outlier measure, with per-parameter gradient checks;
relative errors near zero alone are misleading. Set and document the acceptance
budget from reference behavior and output rounding **before inspecting candidate
error**, then exercise representative inputs and lengths. For example, the
pinned biological tests use twice a native-to-FP32 error envelope plus one
BF16 rounding allowance, for both RMSE and maximum error. For AMPLIFY
packing, that envelope includes an independent PyTorch packed implementation
as well as the original separate GEMMs. Validate the alternative formulation
against separate higher-precision operations before using it to characterize
rounding. Keep the converted implementation out of baseline construction.
For FP32 packing, use the independently measured packed-versus-separate
PyTorch error and FP32 rounding allowance; keep direct FP32 tolerances for
operations without that transformation. This is a tested policy
for those fixtures, not a universal tolerance for every architecture or dtype.
If the native reference itself is numerically unstable, isolate the operation
or choose a better reference before accepting a model-level comparison.

Verify that the policy rejects meaningful defects such as reversed gate order,
missing gradients, and isolated output corruption. Record errors, budgets, and
the reference construction. Do not increase an arbitrary percentage until a
failing port passes, or narrow conversion scope solely to recover bitwise parity.
Use justified direct tolerances when a higher-precision reference is unavailable,
and report that limitation rather than claiming a calibrated result.

## Completion evidence

Confirm that the selected TE modules are reached during the validation run, using
temporary call instrumentation or the generated integration's focused tests.
Preserve the input representation; sequence packing is a separate change with
its own segment-isolation and loss/mask equivalence checks.

The completion criteria are correct weight mapping, preserved model interfaces,
native-precision output/gradient parity, and working save/reload behavior for the
requested task. Throughput benchmarks, speedup thresholds, quantization recipe
sweeps, and low-precision convergence tests are outside this workflow.

Record exact commands, model and config revisions, shapes/layouts, dtypes,
GPU model/count/topology, framework/TE/CUDA/cuDNN/driver versions,
relevant `NVTE_*` flags, and selected backend. Separate passed, failed, skipped,
and unrun checks. Distinguish an environment blocker from an architectural
mismatch or a numerical defect.
