# Troubleshoot the failing stage

Consult the target release's `docs/troubleshooting.rst` and `docs/envvars.rst`
alongside the first actionable error in the verbose log.

| Evidence | Next check or correction |
|---|---|
| Framework not importable during build | Verify the installer interpreter, install the selected framework there, then build with `--no-build-isolation`. |
| `nvcc` absent or unsupported compiler/toolkit | Check toolkit root and executable paths; provision a compatible development toolkit or use a suitable development container. |
| Missing `cudnn.h`, CUDA headers, or NCCL headers | Locate development files and configure their actual roots; runtime libraries alone are insufficient. Check optional EP requirements before disabling EP. |
| Missing `cudnn_frontend.h` | Check the target release's `nvidia-cudnn-frontend` build dependency and header installation. |
| Compiler killed or out-of-memory failure | Check memory and concurrent jobs, reduce build concurrency, then resume using the same configuration. |
| Undefined symbols on import | Inspect extension origin, framework version, C++ ABI, and linked libraries before rebuilding against the active framework. |
| `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` | Check for competing system and environment cuDNN installations; align build and runtime library resolution. |
| JAX custom call has no registered CUDA implementation | Confirm the JAX extension was built and installed in this environment with `--no-build-isolation` and compatible JAX dependencies. |
| Import resolves to another checkout/package | Check interpreter, working directory, editable-install metadata, and `PYTHONPATH`; verify from outside the checkout. |
| No GPU visible | Compare framework and host probes; retry sandbox failures through the harness's permission mechanism. Do not infer a driver defect from a sandbox-only failure. |

## Header and library discovery

The inventory lists `nvidia/*/include` and `nvidia/*/lib` directories belonging to
the active interpreter. Inspect the relevant directories; do not indiscriminately
prepend every installed CUDA library to the environment. For a selected wheel
cuDNN installation, set `CUDNN_PATH` (and `CUDNN_HOME` if needed by the chosen
tooling), and prepend its `lib` directory to `LD_LIBRARY_PATH`. Preserve existing
path entries. Apply `CPATH` or `LIBRARY_PATH` only when required header or linker
discovery fails and the chosen toolchain does not already locate those files.

Current TE also documents `NVTE_CUDA_INCLUDE_PATH` for runtime toolkit-header
discovery. It does not replace a missing build toolkit. Verify support in the
target release before using it.

## Reproduce before broad changes

Use verbose installation output to identify the failing command. On Linux,
`ldd` on the printed extension or common library can reveal unresolved or
unexpected dependencies. Avoid upgrading the framework, changing CUDA majors,
or replacing system libraries as a speculative fix. Rebuild against the active
environment when a dependency or native build configuration changed.

If blocked, provide the failing command, first relevant error, target release,
Python/framework/toolkit/driver versions, GPU visibility, artifact paths, and
relevant build flags. Distinguish install success, library-load success, and
GPU-computation success in the report.
