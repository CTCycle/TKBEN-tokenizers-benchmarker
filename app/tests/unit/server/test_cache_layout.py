from __future__ import annotations

import json
from pathlib import Path

from server.common.path import CACHE_PATH, DATASETS_PATH, ROOT_DIR, TOKENIZERS_PATH


def _read(relative_path: str) -> str:
    return (ROOT_DIR / relative_path).read_text(encoding="utf-8")


def test_disposable_cache_root_is_separate_from_persistent_source_data() -> None:
    assert CACHE_PATH == (ROOT_DIR / "runtimes" / "cache").resolve()
    assert DATASETS_PATH.name == "datasets"
    assert TOKENIZERS_PATH.name == "tokenizers"
    assert DATASETS_PATH.parent.name == "sources"
    assert TOKENIZERS_PATH.parent.name == "sources"
    assert CACHE_PATH not in DATASETS_PATH.parents
    assert CACHE_PATH not in TOKENIZERS_PATH.parents


def test_pytest_temp_path_uses_the_canonical_root(tmp_path: Path) -> None:
    assert CACHE_PATH in tmp_path.parents
    assert tmp_path.parts[: len(CACHE_PATH.parts)] == CACHE_PATH.parts


def test_tooling_configuration_points_to_the_canonical_cache_root() -> None:
    pytest_config = _read("app/tests/pytest.ini")
    angular_config = json.loads(_read("app/client/angular.json"))
    npm_config = _read("app/client/.npmrc")
    ruff_config = _read("app/server/pyproject.toml")
    uv_config = _read("app/server/uv.toml")
    launcher = _read("start_on_windows.ps1")
    runner = _read("app/tests/run_tests.bat")
    ci = _read(".github/workflows/ci.yml")

    assert "cache_dir = ../../runtimes/cache/pytest-state" in pytest_config
    assert angular_config["cli"]["cache"]["path"] == "../../runtimes/cache/angular"
    assert "cache=../../runtimes/cache/npm" in npm_config
    assert 'cache-dir = "../../runtimes/cache/ruff"' in ruff_config
    assert 'cache-dir = "../../runtimes/cache/uv"' in uv_config
    assert "$ToolCacheDir" not in launcher
    assert "Join-Path $TestsDir 'cache'" not in launcher
    assert "%TESTS_DIR%\\cache" not in runner
    assert "app/tests/cache" not in ci
    assert "enable-cache: true" not in ci
    assert "cache: npm" not in ci

    assert "$env:UV_CACHE_DIR = $UvCacheDir" in launcher
    for cache_name in (
        "pip",
        "npm",
        "ruff",
        "mypy",
        "pycache",
        "coverage",
        "playwright",
        "pytest-state",
        "pytest-basetemp-current",
        "angular",
        "matplotlib",
    ):
        assert f"runtimes/cache/{cache_name}" in ci or cache_name in runner


def test_persistent_source_paths_are_not_reclassified_as_cache_constants() -> None:
    path_module = _read("app/server/common/path.py")

    assert "DATASETS_PATH = SOURCES_PATH / \"datasets\"" in path_module
    assert "TOKENIZERS_PATH = SOURCES_PATH / \"tokenizers\"" in path_module
    assert "DATASET_CACHE_PATH" not in path_module
    assert "TOKENIZER_CACHE_PATH" not in path_module
