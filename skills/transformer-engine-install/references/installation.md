# Installation routes

Resolve all versions and requirements against the requested TE release before
running these examples. Commands assume the selected environment is active.

## Released package

Install the requested GPU-enabled PyTorch or JAX stack using that framework's
official instructions first. Confirm it can execute on the intended NVIDIA GPU.
For JAX, include the Flax and other dependencies required by the target TE release.

Choose a TE release compatible with that stack, then replace `X.Y.Z` below:

```bash
python -m pip install --no-build-isolation 'transformer_engine[pytorch]==X.Y.Z'
```

Use `[jax]` or `[pytorch,jax]` for the corresponding frontends. With uv, the
equivalent is `uv pip install --python /path/to/python --no-build-isolation ...`.
TE ships core-library wheels and source distributions for framework extensions;
a PyPI install may still compile native code. Provision build dependencies from
the target release's build metadata before using `--no-build-isolation`.

For common-only consumers, consult the release's core extras. Current source
defines `[core]`, `[core_cu12]`, and `[core_cu13]`; availability and CUDA-major
selection must be checked for older releases. The bare package does not imply
that the requested framework extension is installed.

## Source checkout

Use the existing requested checkout or clone the requested ref. Do not switch an
existing working tree to another release merely to follow this example.

```bash
git submodule update --init --recursive
NVTE_FRAMEWORK=pytorch python -m pip install -e . -v --no-build-isolation
```

Run from the repository root. Select `jax`, `pytorch,jax`, or `none` when needed.
Install the selected frameworks before the TE build. Inspect `pyproject.toml`
and relevant `build_tools/` files for build dependencies; install only what the
chosen configuration needs. A core wheel does not require a framework, but a
common-only source build still needs the native toolchain and build dependencies.

Before compiling:

- Locate `nvcc` and the toolkit root (`CUDA_HOME`/`CUDA_PATH`). A working driver
  or a framework CUDA runtime does not establish that a toolkit is installed.
  Compare `nvcc --version` with framework CUDA information and release support;
  the CUDA version displayed by `nvidia-smi` is a driver capability indicator.
- Confirm cuDNN headers and libraries resolve to one compatible installation.
  When supplied by wheels, use the inventory's paths as candidates and check
  that required headers actually exist. Set `CUDNN_PATH` to the chosen root.
- Inspect CPU count and available memory. Respect existing `MAX_JOBS`,
  `NVTE_BUILD_MAX_JOBS`, and `NVTE_BUILD_THREADS_PER_JOB`; set appropriate limits
  for this machine when needed. Do not reset them to a fixed default.
- For a local development build, select the exact tested GPU architecture in
  `NVTE_CUDA_ARCHS`. Use architecture-specific Blackwell targets such as `100a`
  where supported by the target release/toolkit. A family target such as `100`
  can expand into additional variants. Inspect `transformer_engine/common/CMakeLists.txt` and
  `build_tools/utils.py`; do not generate a universal mapping from GPU names.
  Cross-compilation and portable builds need the intended target list rather
  than the first visible GPU's capability.
- Check optional features. Current source may enable NCCL EP for applicable
  Hopper-or-newer targets. If NCCL development files are missing, provision them
  when EP is required. If EP is outside the requested build, explicitly set
  `NVTE_WITH_NCCL_EP=0` and record the reduced feature set. Likewise preserve
  requested MPI, NVSHMEM, cuBLASMp, cuSolverMp, and CuTeDSL settings.

Keep the selected configuration for subsequent rebuilds. Inspect
`NVTE_CMAKE_BUILD_DIR` and build logs before clearing generated state; a changed
toolchain, generator, or incompatible feature configuration may need a separate
clean build directory. Avoid deleting unrelated build artifacts.

## Verification boundaries

Use the bundled smoke script after the install command succeeds. Retain the log
and command exit status if capturing output (`set -o pipefail` with `tee` in
Bash). An editable Python import can still load stale native libraries, so
inspect every printed artifact path and rebuild when those artifacts are stale.

For common-only installs:

```bash
python -c 'import transformer_engine.common as te; print(te.__file__)'
```

This proves library loading only. Native operator validation follows the
target checkout's `tests/cpp/` and `qa/L0_cppunittest/test.sh` instructions.
