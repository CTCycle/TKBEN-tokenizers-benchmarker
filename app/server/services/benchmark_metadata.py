from __future__ import annotations

import importlib.metadata
import os
import platform
import sys

import psutil


TRACKED_PACKAGES = [
    "tokenizers",
    "transformers",
    "tiktoken",
    "sentencepiece",
    "numpy",
    "pandas",
    "fastapi",
]


def collect_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in TRACKED_PACKAGES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def collect_runtime_environment() -> dict[str, object]:
    env_keys = [
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "PYTHONHASHSEED",
    ]
    try:
        physical_cores = psutil.cpu_count(logical=False)
    except Exception:
        physical_cores = None
    try:
        total_memory_mb = float(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        total_memory_mb = None
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cpu_count_physical": physical_cores,
        "memory_total_mb": total_memory_mb,
        "package_versions": collect_package_versions(),
        "environment": {key: os.environ.get(key, "") for key in env_keys},
    }
