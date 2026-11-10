---
name: transformer-engine-install
description: "Install, build, and troubleshoot NVIDIA Transformer Engine (TE) for PyTorch, JAX, or the common C++ library. Use for pip or uv installs, editable source builds, missing CUDA/cuDNN/NCCL headers, ABI or shared-library import errors, and GPU installation smoke tests."
---

# Transformer Engine Installation

## When to Use This Skill

Use for TE environment setup, native builds, and installation failures. The end
state is the requested TE version and frontend loading from the intended location
and executing a checked GPU operation. Model conversion, precision tuning, and
training-system setup are separate tasks.

## Workflow

### 1. Establish the target and inspect the environment

Determine the requested release or source checkout, frontend (`pytorch`, `jax`,
both, or common-only), Python environment, and container or host context. Infer
these from the request and environment; ask only for consequential missing choices.
Preserve an existing working framework installation and explicit build settings.

Use the target release's `README.rst`, `docs/installation.rst`,
`docs/troubleshooting.rst`, and `pyproject.toml` as compatibility authority. In a
checkout, also read its `AGENTS.md` when present. For installed releases without
source, use the matching tag or versioned official documentation. Do not apply
current-main requirements to an older release or copy historical version tables.

Run the bundled inventory with the Python interpreter that will install TE:

```bash
python /path/to/skill/scripts/inspect_environment.py
```

The script reports versions, CUDA tools, visible GPUs, build settings, and NVIDIA
wheel header/library directories without importing frameworks or installing
packages. It is an inventory, not a compatibility verdict. Check framework GPU
visibility separately (`torch.cuda.is_available()` or `jax.devices()`). A failed
sandbox GPU probe is inconclusive: retry outside the sandbox using the harness's
permission mechanism before diagnosing absent hardware or a broken driver.

### 2. Select and execute the installation route

| Target | Route |
|---|---|
| Suitable NGC image with TE already installed | Verify the existing version and requested frontend first; rebuild only when needed. |
| Released Python package | Install a chosen release with explicit frontend extras, using the active interpreter. |
| Local modifications, fork, or PR | Build the intended checkout; use an editable install for development. |
| C/C++ consumer without a Python frontend | Use the target release's core wheel or `NVTE_FRAMEWORK=none` source build. |

Read [installation.md](references/installation.md) for commands and source-build
decisions. Install the selected GPU framework before building its TE extension and
use `--no-build-isolation`. Core wheels and framework extensions have different
build requirements; do not require a compiler merely to inspect a prebuilt install.
Use `python -m pip` or `uv pip --python` consistently with the selected interpreter.
Do not introduce Megatron-LM, its extras, or its dependency-lock workflow.

Native builds can take substantial time. Keep logs and preserve the install exit
status; a quiet build alone is not evidence of a hang. Choose concurrency from
available CPUs and memory rather than a fixed machine-independent job count.

### 3. Diagnose failures using evidence

Read [troubleshooting.md](references/troubleshooting.md) when a prerequisite,
build, import, or execution check fails. Identify the failing stage and first
actionable error before changing dependencies. After a targeted correction,
rerun that stage. If the same failure persists without new evidence, report the
blocker and required prerequisite instead of repeatedly reinstalling packages.

### 4. Verify the installation

Run from a directory outside the source checkout to avoid accidental source
shadowing. Use the installed environment's Python and an absolute script path:

```bash
python /path/to/skill/scripts/smoke_test.py --framework pytorch
python /path/to/skill/scripts/smoke_test.py --framework jax
```

Run only requested frontends, each in a fresh process. For an editable install,
append `--expected-source /absolute/path/to/TransformerEngine` to check Python
package provenance. Inspect the printed extension and common-library paths too;
the Python path check alone cannot establish native-artifact provenance.

The smoke test checks FP32 LayerNorm output and input gradients against a native
framework reference on one GPU. It does not establish FP8/FP4 support, distributed
correctness, or performance. For JAX, run with `NVTE_JAX_CUSTOM_CALLS` and
`NVTE_JAX_CUSTOM_CALLS_RE` unset so disabled custom calls cannot produce a
fallback-only pass. For a target release with a different API, adapt
the check to that release's documented interface rather than upgrading it silently.
For common-only installs, verify `import transformer_engine.common` and library
location; report that this is a load check, not a GPU computation test. Use the
repository's focused C++ tests if native computation validation is requested.

Report the exact install and verification commands, TE/framework/CUDA versions,
GPU, import and extension paths, and any untested frontend or blocked check.
Never describe an import-only or CPU-only result as a successful GPU install.
