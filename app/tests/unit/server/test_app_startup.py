from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from server import app as app_module
from server.services import startup_validation

###############################################################################
def test_build_cors_origins_normalizes_local_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UI_HOST", "0.0.0.0")
    monkeypatch.setenv("UI_PORT", "8000")

    origins = startup_validation.build_cors_origins()

    assert origins == ["http://127.0.0.1:8000", "http://localhost:8000"]

###############################################################################
def test_build_cors_origins_rejects_invalid_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UI_PORT", "invalid")

    with pytest.raises(RuntimeError, match="UI_PORT must be a valid integer"):
        startup_validation.build_cors_origins()

###############################################################################
def test_run_startup_validations_loads_environment_and_creates_runtime_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    logs_path = tmp_path / "logs"
    datasets_path = tmp_path / "datasets"
    tokenizers_path = tmp_path / "tokenizers"
    templates_path = tmp_path / "templates"

    monkeypatch.setattr(
        startup_validation,
        "ensure_environment_loaded",
        lambda: calls.append("environment"),
    )
    monkeypatch.setattr(startup_validation, "LOGS_PATH", logs_path)
    monkeypatch.setattr(startup_validation, "DATASETS_PATH", datasets_path)
    monkeypatch.setattr(startup_validation, "TOKENIZERS_PATH", tokenizers_path)
    monkeypatch.setattr(startup_validation, "TEMPLATES_PATH", templates_path)

    startup_validation.run_startup_validations()

    assert calls == ["environment"]
    assert logs_path.is_dir()
    assert datasets_path.is_dir()
    assert tokenizers_path.is_dir()
    assert templates_path.is_dir()

###############################################################################
def test_create_app_initializes_startup_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    settings = SimpleNamespace(
        database=SimpleNamespace(embedded_database=True),
        jobs=SimpleNamespace(terminal_retention_seconds=3600.0),
    )

    monkeypatch.setattr(app_module, "get_server_settings", lambda: settings)
    monkeypatch.setattr(
        app_module,
        "run_startup_validations",
        lambda **kwargs: calls.append("validated"),
    )
    monkeypatch.setattr(
        app_module,
        "initialize_database",
        lambda **kwargs: calls.append(f"database:{kwargs['startup']}"),
    )

    application = app_module.create_app()

    with TestClient(application) as client:
        response = client.get("/")

    assert response.status_code in {200, 307}
    assert application.state.settings is settings
    assert calls == ["validated", "database:True"]
