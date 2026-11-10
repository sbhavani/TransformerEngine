# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Check TE LayerNorm forward and input gradients on one GPU in native precision."""

import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import sys


def report_origin(framework, expected_source):
    """Print Python and native artifact locations before numerical checks."""
    te = importlib.import_module("transformer_engine")
    origin = Path(te.__file__).resolve()
    print("python:", sys.executable, flush=True)
    print("TE package:", origin, flush=True)
    print("TE version:", importlib.metadata.version("transformer-engine"), flush=True)
    if expected_source and not origin.is_relative_to(
        Path(expected_source).resolve() / "transformer_engine"
    ):
        raise RuntimeError(f"TE Python package does not originate from {expected_source}")
    importlib.import_module(f"transformer_engine.{framework}")
    extension = importlib.import_module(
        "transformer_engine_torch" if framework == "pytorch" else "transformer_engine_jax"
    )
    print("TE extension:", extension.__file__, flush=True)
    common = importlib.import_module("transformer_engine.common")
    handle = getattr(common, "_TE_LIB_CTYPES", None)
    print("TE common library:", getattr(handle, "_name", "unavailable in this release"), flush=True)


def check_pytorch():
    """Compare TE and native FP32 LayerNorm with a nonuniform upstream gradient."""
    import torch
    import transformer_engine.pytorch as te

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA GPU")
    print("torch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
    print("cuDNN:", torch.backends.cudnn.version(), flush=True)
    x = torch.sin(torch.arange(512, device="cuda", dtype=torch.float32)).reshape(4, 128)
    x.requires_grad_(True)
    reference_x = x.detach().clone().requires_grad_(True)
    layer = te.LayerNorm(128, eps=1e-5, dtype=torch.float32, device="cuda")
    output = layer(x)
    reference = torch.nn.functional.layer_norm(
        reference_x, (128,), layer.weight.detach(), layer.bias.detach(), eps=1e-5
    )
    upstream = torch.cos(x.detach())
    gradient = torch.autograd.grad(output, x, upstream)[0]
    reference_gradient = torch.autograd.grad(reference, reference_x, upstream)[0]
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(gradient, reference_gradient, rtol=1e-4, atol=1e-5)


def check_jax():
    """Compare jitted TE Flax LayerNorm and its gradient with native JAX."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from transformer_engine.jax import flax as te

    # A fallback must not turn an extension smoke test into a native-only pass.
    for setting in ("NVTE_JAX_CUSTOM_CALLS", "NVTE_JAX_CUSTOM_CALLS_RE"):
        if os.environ.get(setting) is not None:
            raise RuntimeError(f"Run this extension smoke test with {setting} unset")
    device = jax.devices("gpu")[0]
    print("jax:", jax.__version__, "GPU:", device, device.device_kind, flush=True)
    print("jaxlib:", importlib.metadata.version("jaxlib"), flush=True)
    with jax.default_device(device):
        x = jnp.sin(jnp.arange(512, dtype=jnp.float32)).reshape(4, 128)
        layer = te.LayerNorm(epsilon=1e-5, dtype=jnp.float32)
        variables = layer.init(jax.random.PRNGKey(0), x)
        apply = jax.jit(lambda value: layer.apply(variables, value))

        def reference(value):
            centered = value - jnp.mean(value, axis=-1, keepdims=True)
            return centered * jax.lax.rsqrt(jnp.mean(centered**2, axis=-1, keepdims=True) + 1e-5)

        upstream = jnp.cos(x)
        output = apply(x)
        gradient = jax.jit(jax.grad(lambda value: jnp.sum(apply(value) * upstream)))(x)
        reference_gradient = jax.grad(lambda value: jnp.sum(reference(value) * upstream))(x)
        output.block_until_ready()
        gradient.block_until_ready()
        np.testing.assert_allclose(output, reference(x), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(gradient, reference_gradient, rtol=1e-4, atol=1e-5)


def main():
    """Run exactly the selected frontend and fail on import or numerical errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("pytorch", "jax"), required=True)
    parser.add_argument("--expected-source", type=Path, help="Expected editable TE repository root")
    args = parser.parse_args()
    report_origin(args.framework, args.expected_source)
    if args.framework == "pytorch":
        check_pytorch()
    else:
        check_jax()
    print(f"PASS: {args.framework} GPU LayerNorm FP32 output and input gradients (shape 4x128)")


if __name__ == "__main__":
    main()
