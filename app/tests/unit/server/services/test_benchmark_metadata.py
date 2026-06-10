from __future__ import annotations

from server.services.benchmark_metadata import collect_runtime_environment


###############################################################################
def test_collect_runtime_environment_contains_reproducibility_fields() -> None:
    metadata = collect_runtime_environment()
    assert "python_version" in metadata
    assert "platform" in metadata
    assert "processor" in metadata
    assert "cpu_count" in metadata
    assert "cpu_count_physical" in metadata
    assert "memory_total_mb" in metadata
    assert "package_versions" in metadata
    assert "environment" in metadata

    environment = metadata["environment"]
    assert isinstance(environment, dict)
    for key in [
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "PYTHONHASHSEED",
    ]:
        assert key in environment
