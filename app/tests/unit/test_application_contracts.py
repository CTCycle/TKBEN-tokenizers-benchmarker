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
import ast
from server.app import app
import pytest
from server.common.utils.security import ensure_path_is_within
from server.common.utils.security import normalize_upload_stem
import time
from server.services import jobs as jobs_module
from server.configurations import DatabaseSettings
from server.repositories.database.backend import TKBENDatabase
from server.repositories.database.initializer import initialize_database
from server.repositories.jobs import JobRepository
from server.services.jobs import JobManager

###############################################################################


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

###############################################################################


SERVER_ROOT = Path(__file__).resolve().parents[2] / "server"
REPOSITORY_ROOT = SERVER_ROOT.parents[1]

FORBIDDEN_IMPORTS: dict[str, tuple[str, ...]] = {
    "api": ("server.repositories",),
    "contracts": (
        "server.api",
        "server.services",
        "server.repositories",
        "fastapi",
        "sqlalchemy",
    ),
    "services": ("server.api",),
    "repositories": ("server.api", "server.services"),
    "configurations": ("server.api",),
}

LEGACY_IMPORTS = (
    "server.domain",
    "server.repositories.serialization",
    "server.configurations.management",
    "server.common.utils.types",
)

REMOVED_COMPATIBILITY_PATHS = (
    SERVER_ROOT / "domain",
    SERVER_ROOT / "repositories" / "serialization.py",
    SERVER_ROOT / "configurations" / "management.py",
    SERVER_ROOT / "common" / "utils" / "types.py",
)

###############################################################################
def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return modules

###############################################################################
def _production_python_files() -> list[Path]:
    return sorted(
        path
        for path in SERVER_ROOT.rglob("*.py")
        if not any(part in {".venv", "__pycache__"} for part in path.parts)
    )

###############################################################################
def _starts_with_module(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")

###############################################################################
def test_configuration_bootstraps_before_sensitive_imports() -> None:
    configuration_init = SERVER_ROOT / "configurations" / "__init__.py"
    tree = ast.parse(
        configuration_init.read_text(encoding="utf-8"),
        filename=str(configuration_init),
    )

    bootstrap_index = next(
        index
        for index, node in enumerate(tree.body)
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "ensure_environment_loaded"
        )
    )
    sensitive_import_indices = [
        index
        for index, node in enumerate(tree.body)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            in {
                "server.configurations.settings",
                "server.configurations.startup",
            }
        )
    ]

    assert sensitive_import_indices
    assert bootstrap_index < min(sensitive_import_indices)

###############################################################################
def test_production_layers_respect_dependency_boundaries() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        relative = path.relative_to(SERVER_ROOT)
        layer = relative.parts[0] if len(relative.parts) > 1 else ""
        forbidden = FORBIDDEN_IMPORTS.get(layer, ())
        for module in _imported_modules(path):
            for prefix in forbidden:
                if _starts_with_module(module, prefix):
                    violations.append(f"{relative}: imports {module}")

    assert not violations, "\n".join(violations)

###############################################################################
def test_production_code_has_no_legacy_architecture_imports() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        relative = path.relative_to(SERVER_ROOT)
        for module in _imported_modules(path):
            if any(_starts_with_module(module, prefix) for prefix in LEGACY_IMPORTS):
                violations.append(f"{relative}: imports {module}")

    assert not violations, "\n".join(violations)

###############################################################################
def test_removed_compatibility_paths_do_not_return() -> None:
    existing = [str(path.relative_to(REPOSITORY_ROOT)) for path in REMOVED_COMPATIBILITY_PATHS if path.exists()]
    assert not existing, "Removed compatibility paths were recreated:\n" + "\n".join(existing)

###############################################################################


###############################################################################
JSON_ROUTE_EXPECTATIONS = [
    ("/api/health", "get", 200, "HealthResponse"),
    ("/api/datasets/list", "get", 200, "DatasetListResponse"),
    ("/api/datasets/metrics/catalog", "get", 200, "DatasetMetricCatalogResponse"),
    ("/api/datasets/download", "post", 202, "JobStartResponse"),
    ("/api/datasets/upload", "post", 202, "JobStartResponse"),
    ("/api/datasets/analyze", "post", 202, "JobStartResponse"),
    ("/api/datasets/reports/latest", "get", 200, "DatasetAnalysisResponse"),
    ("/api/datasets/reports/{report_id}", "get", 200, "DatasetAnalysisResponse"),
    ("/api/datasets/delete", "delete", 200, "DatasetDeleteResponse"),
    ("/api/settings", "get", 200, "RuntimeSettingsResponse"),
    ("/api/settings", "patch", 200, "RuntimeSettingsResponse"),
    ("/api/settings/reset", "post", 200, "RuntimeSettingsResponse"),
    ("/api/tokenizers/discover", "get", 200, "TokenizerDiscoveryResponse"),
    ("/api/tokenizers/list", "get", 200, "TokenizerListResponse"),
    ("/api/tokenizers/download", "post", 202, "JobStartResponse"),
    ("/api/tokenizers/reports/generate", "post", 202, "JobStartResponse"),
    ("/api/tokenizers/reports/latest", "get", 200, "TokenizerReportResponse"),
    ("/api/tokenizers/reports/{report_id}", "get", 200, "TokenizerReportResponse"),
    (
        "/api/tokenizers/reports/{report_id}/vocabulary",
        "get",
        200,
        "TokenizerVocabularyPageResponse",
    ),
    ("/api/tokenizers/upload", "post", 200, "TokenizerUploadResponse"),
    ("/api/tokenizers/preview", "post", 200, "TokenizerPreviewResponse"),
    ("/api/tokenizers/delete", "delete", 200, "TokenizerDeleteResponse"),
    ("/api/benchmarks/run", "post", 202, "JobStartResponse"),
    ("/api/benchmarks/reports", "get", 200, "BenchmarkReportListResponse"),
    ("/api/benchmarks/reports/{report_id}", "get", 200, "BenchmarkRunResponse"),
    (
        "/api/benchmarks/reports/{report_id}/tags",
        "patch",
        200,
        "BenchmarkReportTagsResponse",
    ),
    (
        "/api/benchmarks/metrics/catalog",
        "get",
        200,
        "BenchmarkMetricCatalogResponse",
    ),
    ("/api/jobs/{job_id}", "get", 200, "JobStatusResponse"),
    ("/api/jobs/{job_id}/cancel", "post", 200, "JobStatusResponse"),
    ("/api/keys", "post", 201, "HFAccessKeyListItem"),
    ("/api/keys", "get", 200, "HFAccessKeyListResponse"),
    ("/api/keys/{key_id}", "delete", 200, "HFAccessKeyDeleteResponse"),
    ("/api/keys/{key_id}/activate", "post", 200, "HFAccessKeyActivateResponse"),
    ("/api/keys/{key_id}/deactivate", "post", 200, "HFAccessKeyActivateResponse"),
    ("/api/keys/{key_id}/reveal", "post", 200, "HFAccessKeyRevealResponse"),
]

###############################################################################
def test_openapi_generation_and_response_models() -> None:
    schema = app.openapi()
    assert schema

    paths = schema.get("paths", {})
    assert "/api/tokenizers/custom" not in paths
    for path, method, status_code, model_name in JSON_ROUTE_EXPECTATIONS:
        assert path in paths
        operation = paths[path][method]
        content = operation["responses"][str(status_code)]["content"][
            "application/json"
        ]["schema"]
        assert content.get("$ref", "").endswith(f"/{model_name}")

    assert "delete" in paths["/api/benchmarks/reports/{report_id}"]
    assert "204" in paths["/api/benchmarks/reports/{report_id}"]["delete"]["responses"]
    components = schema.get("components", {}).get("schemas", {})
    assert "RuntimeSettingsPatchRequest" in components
    assert "RuntimeSettingsResetRequest" in components
    assert components["RuntimeSettingsPatchRequest"]["additionalProperties"] is False

###############################################################################


###############################################################################
def test_normalize_upload_stem_handles_windows_style_names() -> None:
    assert normalize_upload_stem(r"nested\folder\Sample File.csv") == "Sample_File"

###############################################################################
def test_normalize_upload_stem_handles_posix_style_names() -> None:
    assert normalize_upload_stem("nested/folder/report.final.xlsx") == "report.final"

###############################################################################
def test_ensure_path_is_within_accepts_str_and_path_inputs(tmp_path: Path) -> None:
    base_path = tmp_path / "datasets"
    candidate_path = base_path / "safe" / "artifact.json"

    resolved = ensure_path_is_within(str(base_path), candidate_path)

    assert resolved == str(candidate_path.resolve())

###############################################################################
def test_ensure_path_is_within_rejects_escape_attempts(tmp_path: Path) -> None:
    base_path = tmp_path / "datasets"
    candidate_path = base_path / ".." / "outside.txt"

    with pytest.raises(ValueError, match="escapes the allowed base directory"):
        ensure_path_is_within(base_path, candidate_path)

###############################################################################


###############################################################################
def _wait_for_status(
    manager: JobManager,
    job_id: str,
    expected_status: str,
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = manager.get_job_status(job_id)
        if status and status["status"] == expected_status:
            return status
        time.sleep(0.01)
    status = manager.get_job_status(job_id)
    raise AssertionError(f"Expected {expected_status}, got {status}")

###############################################################################
def test_completed_job_remains_visible_within_retention() -> None:
    manager = JobManager(terminal_retention_seconds=60.0)

    job_id = manager.start_job("sample", lambda: {"value": 1})

    status = _wait_for_status(manager, job_id, "completed")

    assert status["result"] == {"value": 1}
    assert manager.get_job_status(job_id) is not None

###############################################################################
def test_terminal_jobs_are_pruned_after_retention() -> None:
    current_time = {"value": 100.0}
    manager = JobManager(
        terminal_retention_seconds=5.0,
        clock=lambda: current_time["value"],
    )

    job_id = manager.start_job("sample", lambda: {"value": 1})
    status = _wait_for_status(manager, job_id, "completed")
    assert status["completed_at"] == 100.0

    current_time["value"] = 106.0

    assert manager.get_job_status(job_id) is None

###############################################################################
def test_running_job_can_be_stopped_cooperatively() -> None:
    manager = JobManager(terminal_retention_seconds=60.0)

    def runner(job_id: str) -> dict[str, object]:
        while not manager.should_stop(job_id):
            time.sleep(0.01)
        return {"ignored": True}

    job_id = manager.start_job("sample", runner)

    stop_status = manager.request_stop(job_id)
    assert stop_status is not None
    assert stop_status["status"] == "running"
    assert manager.should_stop(job_id) is True

    status = _wait_for_status(manager, job_id, "cancelled")
    assert status["result"] is None

###############################################################################
def test_stopping_unknown_job_returns_none() -> None:
    assert JobManager().request_stop("missing") is None

###############################################################################
def _job_repository(tmp_path: Path) -> tuple[JobRepository, TKBENDatabase]:
    settings = DatabaseSettings(
        embedded_database=True,
        sqlite_path=tmp_path / "jobs.db",
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=5,
        insert_batch_size=1000,
    )
    database = TKBENDatabase(settings)
    initialize_database(settings=settings)
    return JobRepository(database), database

###############################################################################
def test_terminal_job_history_survives_manager_recreation(tmp_path: Path) -> None:
    repository, database = _job_repository(tmp_path)
    manager = JobManager(repository=repository)
    try:
        completed_id = manager.start_job("sample", lambda: {"value": "saved"})

        def fail(job_id: str) -> dict[str, object]:
            manager.update_result(job_id, {"phase": "started"})
            raise ValueError("runner failed")

        failed_id = manager.start_job("sample", fail)

        def wait_for_stop(job_id: str) -> dict[str, object]:
            manager.update_result(job_id, {"phase": "waiting"})
            while not manager.should_stop(job_id):
                time.sleep(0.01)
            return {"discarded": True}

        cancelled_id = manager.start_job("sample", wait_for_stop)
        manager.request_stop(cancelled_id)
        _wait_for_status(manager, completed_id, "completed")
        _wait_for_status(manager, failed_id, "failed")
        _wait_for_status(manager, cancelled_id, "cancelled")

        restored_manager = JobManager(repository=repository)
        restored_manager.restore_persisted_jobs()

        completed = restored_manager.get_job_status(completed_id)
        failed = restored_manager.get_job_status(failed_id)
        cancelled = restored_manager.get_job_status(cancelled_id)
        assert completed is not None
        assert completed["result"] == {"value": "saved"}
        assert completed["completed_at"] is not None
        assert failed is not None
        assert failed["error"] == "runner failed"
        assert failed["failure_reason"] == "runner_error"
        assert failed["result"] == {"phase": "started"}
        assert cancelled is not None
        assert cancelled["status"] == "cancelled"
        assert cancelled["result"] == {"phase": "waiting"}
    finally:
        database.backend.engine.dispose()

###############################################################################
def test_restart_reconciliation_fails_pending_and_running_jobs(tmp_path: Path) -> None:
    repository, database = _job_repository(tmp_path)
    now = time.time()
    active_rows = (
        ("pending1", "pending", 0.0),
        ("running1", "running", 37.5),
    )
    try:
        for job_id, status, progress in active_rows:
            repository.save_job(
                {
                    "job_id": job_id,
                    "job_type": "dataset_analysis",
                    "status": status,
                    "progress": progress,
                    "result": {"partial": True},
                    "error": None,
                    "failure_reason": None,
                    "created_at": now - 10,
                    "completed_at": None,
                }
            )

        manager = JobManager(repository=repository)
        manager.restore_persisted_jobs()

        for job_id, status, progress in active_rows:
            snapshot = manager.get_job_status(job_id)
            assert snapshot is not None
            assert snapshot["status"] == "failed"
            assert snapshot["progress"] == progress
            assert snapshot["result"] == {"partial": True}
            assert snapshot["error"] == jobs_module.RESTART_INTERRUPTION_ERROR
            assert snapshot["failure_reason"] == "application_restart"
            assert snapshot["completed_at"] is not None
            stored = repository.get_job(job_id)
            assert stored is not None
            assert stored["failure_reason"] == "application_restart"
    finally:
        database.backend.engine.dispose()

###############################################################################
def test_restart_prunes_expired_persisted_terminal_jobs(tmp_path: Path) -> None:
    repository, database = _job_repository(tmp_path)
    now = time.time()
    try:
        repository.save_job(
            {
                "job_id": "expired1",
                "job_type": "benchmark",
                "status": "completed",
                "progress": 100.0,
                "result": {"report_id": "old"},
                "error": None,
                "failure_reason": None,
                "created_at": now - 30,
                "completed_at": now - 10,
            }
        )
        repository.save_job(
            {
                "job_id": "recent01",
                "job_type": "benchmark",
                "status": "failed",
                "progress": 15.0,
                "result": None,
                "error": "recent failure",
                "failure_reason": "runner_error",
                "created_at": now - 10,
                "completed_at": now - 1,
            }
        )

        manager = JobManager(
            terminal_retention_seconds=5.0,
            repository=repository,
            clock=lambda: now,
        )
        manager.restore_persisted_jobs()

        assert repository.get_job("expired1") is None
        assert manager.get_job_status("expired1") is None
        recent = manager.get_job_status("recent01")
        assert recent is not None
        assert recent["error"] == "recent failure"
    finally:
        database.backend.engine.dispose()
