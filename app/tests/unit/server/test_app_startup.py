from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from server import app as app_module
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
) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(
            logs=logs,
            datasets=datasets,
            tokenizers=tokenizers,
            templates=templates,
        ),
        network=SimpleNamespace(ui_host=ui_host, ui_port=ui_port),
        database=SimpleNamespace(embedded_database=True),
        jobs=SimpleNamespace(terminal_retention_seconds=3600.0),
    )


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

    application = app_module.create_app()

    with TestClient(application) as client:
        response = client.get("/")

    assert response.status_code in {200, 307}
    assert application.state.settings is settings
    assert calls == [
        f"logging:{settings.paths.logs}",
        "validated:True",
        "database:True",
    ]
