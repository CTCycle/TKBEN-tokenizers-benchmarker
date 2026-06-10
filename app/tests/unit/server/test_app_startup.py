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
def test_run_startup_validations_creates_runtime_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("FASTAPI_HOST=127.0.0.1\n", encoding="utf-8")

    logs_path = tmp_path / "logs"
    datasets_path = tmp_path / "datasets"
    tokenizers_path = tmp_path / "tokenizers"
    templates_path = tmp_path / "templates"
    index_path = tmp_path / "client" / "dist" / "index.html"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(startup_validation, "ENV_FILE_PATH", env_path)
    monkeypatch.setattr(startup_validation, "LOGS_PATH", logs_path)
    monkeypatch.setattr(startup_validation, "DATASETS_PATH", datasets_path)
    monkeypatch.setattr(startup_validation, "TOKENIZERS_PATH", tokenizers_path)
    monkeypatch.setattr(startup_validation, "TEMPLATES_PATH", templates_path)

    startup_validation.run_startup_validations(
        tauri_mode_enabled=True,
        client_index_file_path=index_path,
    )

    assert logs_path.is_dir()
    assert datasets_path.is_dir()
    assert tokenizers_path.is_dir()
    assert templates_path.is_dir()


###############################################################################
def test_run_startup_validations_requires_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(startup_validation, "ENV_FILE_PATH", tmp_path / ".env")

    with pytest.raises(RuntimeError, match="Environment file not found"):
        startup_validation.run_startup_validations(
            tauri_mode_enabled=False,
            client_index_file_path=tmp_path / "index.html",
        )


###############################################################################
def test_create_app_initializes_startup_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    settings = SimpleNamespace(database=SimpleNamespace(embedded_database=True))

    monkeypatch.setattr(app_module, "get_server_settings", lambda: settings)
    monkeypatch.setattr(
        app_module,
        "run_startup_validations",
        lambda **kwargs: calls.append("validated"),
    )
    monkeypatch.setattr(
        app_module,
        "initialize_database",
        lambda: calls.append("database"),
    )

    application = app_module.create_app()

    with TestClient(application) as client:
        response = client.get("/")

    assert response.status_code in {200, 307}
    assert application.state.settings is settings
    assert calls == ["validated", "database"]


###############################################################################
def test_create_app_redirects_root_in_web_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TKBEN_TAURI_MODE", "false")
    monkeypatch.setattr(app_module, "get_server_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(app_module, "run_startup_validations", lambda **kwargs: None)
    monkeypatch.setattr(app_module, "initialize_database", lambda: None)

    application = app_module.create_app()

    with TestClient(application) as client:
        response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


###############################################################################
def test_create_app_serves_spa_in_tauri_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dist_path = tmp_path / "dist"
    assets_path = dist_path / "assets"
    assets_path.mkdir(parents=True, exist_ok=True)
    index_path = dist_path / "index.html"
    index_path.write_text("<html><body>tkben</body></html>", encoding="utf-8")
    asset_file = assets_path / "main.js"
    asset_file.write_text("console.log('tkben');", encoding="utf-8")

    monkeypatch.setenv("TKBEN_TAURI_MODE", "true")
    monkeypatch.setattr(app_module, "CLIENT_DIST_PATH", dist_path)
    monkeypatch.setattr(app_module, "CLIENT_ASSETS_PATH", assets_path)
    monkeypatch.setattr(app_module, "CLIENT_INDEX_FILE_PATH", index_path)
    monkeypatch.setattr(app_module, "get_server_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(app_module, "run_startup_validations", lambda **kwargs: None)
    monkeypatch.setattr(app_module, "initialize_database", lambda: None)

    application = app_module.create_app()

    with TestClient(application) as client:
        root_response = client.get("/")
        nested_response = client.get("/dashboards/current")
        asset_response = client.get("/assets/main.js")

    assert root_response.status_code == 200
    assert "tkben" in root_response.text
    assert nested_response.status_code == 200
    assert "tkben" in nested_response.text
    assert asset_response.status_code == 200
    assert "console.log('tkben');" in asset_response.text
