# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Print a read-only TE installation inventory using only the standard library."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig


def command_output(command):
    """Capture bounded tool output; missing tools and failed probes remain evidence."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {"returncode": result.returncode, "output": (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def inventory():
    """Describe this interpreter without importing TE or either frontend."""
    packages = {}
    for name in (
        "transformer-engine",
        "transformer-engine-cu12",
        "transformer-engine-cu13",
        "transformer-engine-torch",
        "transformer-engine-jax",
        "torch",
        "jax",
        "jaxlib",
        "flax",
        "nvidia-cudnn-cu12",
        "nvidia-cudnn-cu13",
        "nvidia-cudnn-frontend",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    settings = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("NVTE_")
        or key in ("CUDA_HOME", "CUDA_PATH", "CUDNN_PATH", "CXX", "MAX_JOBS")
    }
    toolkit = settings.get("CUDA_HOME") or settings.get("CUDA_PATH")
    nvcc = str(Path(toolkit) / "bin/nvcc") if toolkit else "nvcc"
    wheel_paths = sorted(
        {
            str(path)
            for root in sys.path
            if root
            for suffix in ("include", "lib")
            for path in Path(root).glob(f"nvidia/*/{suffix}")
            if path.is_dir()
        }
    )
    disk = shutil.disk_usage(Path.cwd())
    return {
        "note": (
            "Inventory only; compare with target-release requirements and framework GPU probes."
        ),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "prefix": sys.prefix,
        "site_packages": sysconfig.get_path("purelib"),
        "cpu_count": os.cpu_count(),
        "free_disk_bytes": disk.free,
        "packages": packages,
        "build_environment": settings,
        "nvidia_wheel_directories": wheel_paths,
        "tools": {name: shutil.which(name) for name in ("nvcc", "cmake", "ninja", "c++", "git")},
        "nvcc": command_output([nvcc, "--version"]),
        "gpu_probe": command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]
        ),
    }


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))
