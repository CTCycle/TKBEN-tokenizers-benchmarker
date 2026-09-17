from __future__ import annotations

import json
from pathlib import Path

from server.configurations.runtime import RuntimeSettingsStore
from server.configurations.settings import (
    BenchmarkSettings,
    DatabaseSettings,
    DatasetSettings,
    JobsSettings,
    NetworkSettings,
    PathSettings,
    SecuritySettings,
    ServerSettings,
    TokenizerSettings,
)


###############################################################################
def _server_settings(tmp_path: Path) -> ServerSettings:
    resources = tmp_path / "resources"
    return ServerSettings(
        database=DatabaseSettings(
            embedded_database=True,
            sqlite_path=resources / "database.db",
            host=None,
            port=None,
            database_name=None,
            username=None,
            password=None,
            ssl=False,
            ssl_ca=None,
            connect_timeout=30,
            insert_batch_size=1000,
        ),
        paths=PathSettings(
            resources=resources,
            sources=resources / "sources",
            datasets=resources / "sources/datasets",
            tokenizers=resources / "sources/tokenizers",
            logs=resources / "logs",
            templates=resources / "templates",
        ),
        network=NetworkSettings(
            fastapi_host="127.0.0.1",
            fastapi_port=5000,
            ui_host="127.0.0.1",
            ui_port=8000,
            api_base_url="/api",
        ),
        security=SecuritySettings(
            allow_key_reveal=False,
            hf_keys_encryption_material_file=resources / "hf-key-material.json",
        ),
        datasets=DatasetSettings(),
        tokenizers=TokenizerSettings(),
        benchmarks=BenchmarkSettings(),
        jobs=JobsSettings(),
    )


###############################################################################
def test_benchmark_defaults_persist_sparsely_and_reset_independently(tmp_path: Path) -> None:
    path = tmp_path / "runtime-settings.json"
    store = RuntimeSettingsStore(_server_settings(tmp_path), path)

    saved = store.apply_patch(
        {
            "benchmarks": {
                "default_max_documents": 2500,
                "default_batch_size": 32,
                "default_parallelism": 4,
            }
        },
        expected_revision=0,
    )

    assert saved.settings.benchmarks.default_max_documents == 2500
    assert saved.settings.benchmarks.default_batch_size == 32
    assert saved.settings.benchmarks.default_parallelism == 4
    assert saved.overridden_keys == (
        "benchmarks.default_max_documents",
        "benchmarks.default_batch_size",
        "benchmarks.default_parallelism",
    )
    assert json.loads(path.read_text(encoding="utf-8"))["overrides"] == {
        "benchmarks": {
            "default_batch_size": 32,
            "default_max_documents": 2500,
            "default_parallelism": 4,
        }
    }

    reset = store.reset(
        expected_revision=1,
        keys=["benchmarks.default_batch_size"],
    )
    assert reset.settings.benchmarks.default_batch_size == 16
    assert reset.settings.benchmarks.default_max_documents == 2500
    assert reset.settings.benchmarks.default_parallelism == 4
