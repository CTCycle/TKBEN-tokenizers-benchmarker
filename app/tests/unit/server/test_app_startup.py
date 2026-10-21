# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import app as app_module
from server.repositories.database.migrations import DatabaseMigrationError
from server.services import startup_validation

###############################################################################
def _settings(
    *,
    logs: Path,
    datasets: Path,
    tokenizers: Path,
    templates: Path,
    ui_host: str = "127.0.0.1",
    ui_port: int = 8000,
    cors_origins: tuple[str, ...] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(
            data=templates.parent / "data",
            logs=logs,
            datasets=datasets,
            tokenizers=tokenizers,
            templates=templates,
        ),
        network=SimpleNamespace(
            ui_host=ui_host,
            ui_port=ui_port,
            cors_origins=cors_origins,
        ),
        database=SimpleNamespace(embedded_database=True),
        jobs=SimpleNamespace(terminal_retention_seconds=3600.0),
    )


###############################################################################
def test_frontend_routes_serve_build_assets_and_keep_api_routes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frontend_root = tmp_path / "browser"
    assets_root = frontend_root / "assets"
    assets_root.mkdir(parents=True)
    (frontend_root / "index.html").write_text(
        "<!doctype html><html><body>built app</body></html>", encoding="utf-8"
    )
    (assets_root / "main.js").write_text("console.log('built');", encoding="utf-8")
    monkeypatch.setattr(app_module, "FRONTEND_BUILD_DIR", frontend_root)

    application = FastAPI()
    app_module.register_api_routers(application)
    app_module.register_frontend_routes(application)

    with TestClient(application) as client:
        root = client.get("/")
        client_route = client.get("/settings")
        asset = client.get("/assets/main.js")
        docs = client.get("/docs")
        health = client.get("/api/health")
        unknown_api = client.get("/api/not-a-route")
        missing_asset = client.get("/assets/missing.js")

    assert root.status_code == 200
    assert "built app" in root.text
    assert client_route.status_code == 200
    assert client_route.text == root.text
    assert asset.status_code == 200
    assert asset.text == "console.log('built');"
    assert asset.headers["content-type"] in {
        "application/javascript",
        "text/javascript; charset=utf-8",
    }
    assert docs.status_code == 200
    assert "Swagger UI" in docs.text
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert unknown_api.status_code == 404
    assert missing_asset.status_code == 404


###############################################################################
def test_frontend_routes_keep_docs_redirect_when_build_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(app_module, "FRONTEND_BUILD_DIR", tmp_path / "missing-build")
    application = FastAPI()
    app_module.register_frontend_routes(application)

    with TestClient(application, follow_redirects=False) as client:
        response = client.get("/")

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


###############################################################################
def test_frontend_remains_available_in_database_repair_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frontend_root = tmp_path / "browser"
    frontend_root.mkdir()
    (frontend_root / "index.html").write_text("built app", encoding="utf-8")
    settings = _settings(
        logs=tmp_path / "logs",
        datasets=tmp_path / "datasets",
        tokenizers=tmp_path / "tokenizers",
        templates=tmp_path / "templates",
    )
    monkeypatch.setattr(app_module, "FRONTEND_BUILD_DIR", frontend_root)
    monkeypatch.setattr(app_module, "get_server_settings", lambda: settings)
    monkeypatch.setattr(app_module, "configure_logging", lambda path: None)
    monkeypatch.setattr(app_module, "run_startup_validations", lambda received: None)

    def fail_database_initialization(**kwargs) -> None:
        raise DatabaseMigrationError("database unavailable")

    monkeypatch.setattr(app_module, "initialize_database", fail_database_initialization)

    with TestClient(app_module.create_app()) as client:
        frontend = client.get("/settings")
        health = client.get("/api/health")
        database_route = client.get("/api/datasets/list")
        unknown_api = client.get("/api/not-a-route")
        missing_asset = client.get("/missing.js")

    assert frontend.status_code == 200
    assert frontend.text == "built app"
    assert health.status_code == 200
    assert database_route.status_code == 503
    assert unknown_api.status_code == 404
    assert missing_asset.status_code == 404

###############################################################################
def test_build_cors_origins_normalizes_local_hosts(tmp_path: Path) -> None:
    settings = _settings(
        logs=tmp_path / "logs",
        datasets=tmp_path / "datasets",
        tokenizers=tmp_path / "tokenizers",
        templates=tmp_path / "templates",
        ui_host="0.0.0.0",
        ui_port=8000,
    )

    origins = startup_validation.build_cors_origins(settings)

    assert origins == ["http://127.0.0.1:8000", "http://localhost:8000"]

###############################################################################
def test_build_cors_origins_uses_explicit_environment_origins(tmp_path: Path) -> None:
    settings = _settings(
        logs=tmp_path / "logs",
        datasets=tmp_path / "datasets",
        tokenizers=tmp_path / "tokenizers",
        templates=tmp_path / "templates",
        ui_host="0.0.0.0",
        ui_port=8000,
        cors_origins=("https://bench.example", "http://localhost:3000"),
    )

    origins = startup_validation.build_cors_origins(settings)

    assert origins == ["https://bench.example", "http://localhost:3000"]

###############################################################################
def test_run_startup_validations_creates_runtime_directories(tmp_path: Path) -> None:
    logs_path = tmp_path / "logs"
    datasets_path = tmp_path / "datasets"
    tokenizers_path = tmp_path / "tokenizers"
    templates_path = tmp_path / "templates"
    settings = _settings(
        logs=logs_path,
        datasets=datasets_path,
        tokenizers=tokenizers_path,
        templates=templates_path,
    )

    startup_validation.run_startup_validations(settings)

    assert logs_path.is_dir()
    assert datasets_path.is_dir()
    assert tokenizers_path.is_dir()
    assert templates_path.is_dir()

###############################################################################
def test_create_app_initializes_startup_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []
    settings = _settings(
        logs=tmp_path / "logs",
        datasets=tmp_path / "datasets",
        tokenizers=tmp_path / "tokenizers",
        templates=tmp_path / "templates",
        cors_origins=("https://bench.example",),
    )

    monkeypatch.setattr(app_module, "get_server_settings", lambda: settings)
    monkeypatch.setattr(
        app_module,
        "configure_logging",
        lambda path: calls.append(f"logging:{path}"),
    )
    monkeypatch.setattr(
        app_module,
        "run_startup_validations",
        lambda received: calls.append(f"validated:{received is settings}"),
    )
    monkeypatch.setattr(
        app_module,
        "initialize_database",
        lambda **kwargs: calls.append(f"database:{kwargs['startup']}"),
    )
    monkeypatch.setattr(app_module, "get_database", lambda: object())
    monkeypatch.setattr(
        app_module,
        "DatabaseRuntimeSettingsPersistence",
        lambda database: object(),
    )
    runtime_store = object()
    monkeypatch.setattr(
        app_module,
        "build_database_runtime_settings_store",
        lambda received, persistence: runtime_store,
    )
    monkeypatch.setattr(
        app_module,
        "activate_runtime_settings_store",
        lambda received: SimpleNamespace(settings=settings),
    )
    repository = object()
    monkeypatch.setattr(app_module, "JobRepository", lambda: repository)

    ###############################################################################
    class StubJobManager:

        # -------------------------------------------------------------------------
        def __init__(self, terminal_retention_seconds, repository=None) -> None:
            self.repository = repository

        # -------------------------------------------------------------------------
        def restore_persisted_jobs(self) -> None:
            calls.append("jobs:restored")

    monkeypatch.setattr(app_module, "JobManager", StubJobManager)

    application = app_module.create_app()

    with TestClient(application) as client:
        response = client.get("/")
        cors_response = client.options(
            "/api/health",
            headers={
                "Origin": "https://bench.example",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code in {200, 307}
    assert cors_response.status_code == 200
    assert cors_response.headers["access-control-allow-origin"] == (
        "https://bench.example"
    )
    assert application.state.settings is settings
    assert calls == [
        f"logging:{settings.paths.logs}",
        "validated:True",
        "database:True",
        "jobs:restored",
    ]
    assert application.state.job_repository is repository
    assert application.state.job_manager.repository is repository
