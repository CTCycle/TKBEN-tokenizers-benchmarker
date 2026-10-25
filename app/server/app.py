# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.routing import Match

from server.api.benchmarks import router as benchmarks_router
from server.api.datasets import router as datasets_router
from server.api.database import router as database_router
from server.api.exports import router as exports_router
from server.api.jobs import router as jobs_router
from server.api.keys import router as keys_router
from server.api.settings import router as settings_router
from server.api.tokenizers import router as tokenizers_router
from server.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from server.common.utils.logger import configure_logging, logger
from server.configurations import get_server_settings
from server.configurations.startup import (
    activate_runtime_settings_store,
    build_database_runtime_settings_store,
)
from server.contracts.health import HealthResponse
from server.repositories.database.initializer import initialize_database
from server.repositories.database.migrations import DatabaseMigrationError
from server.repositories.jobs import JobRepository
from server.repositories.runtime_settings import DatabaseRuntimeSettingsPersistence
from server.repositories.database.backend import get_database
from server.services.jobs import JobManager
from server.services.database_switch import DatabaseSwitchGate
from server.services.startup_validation import (
    build_cors_origins,
    run_startup_validations,
)

###############################################################################
FRONTEND_BUILD_DIR = (
    Path(__file__).resolve().parents[1]
    / "client"
    / "dist"
    / "tkben-angular"
    / "browser"
)


###############################################################################
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


###############################################################################
def backend_healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


###############################################################################
def register_api_routers(application: FastAPI) -> None:
    application.add_api_route(
        "/api/health",
        backend_healthcheck,
        methods=["GET"],
        response_model=HealthResponse,
    )
    for router in (
        datasets_router,
        tokenizers_router,
        benchmarks_router,
        jobs_router,
        keys_router,
        settings_router,
        database_router,
        exports_router,
    ):
        application.include_router(router, prefix="/api")


###############################################################################
def register_frontend_routes(application: FastAPI) -> None:
    frontend_root = FRONTEND_BUILD_DIR.resolve()
    index_path = frontend_root / "index.html"
    if not index_path.is_file():
        application.add_api_route("/", redirect_to_docs, methods=["GET"])
        return

    def serve_index() -> FileResponse:
        return FileResponse(index_path)

    def serve_frontend_path(path: str) -> FileResponse:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")

        requested_path = (frontend_root / path).resolve()
        if not requested_path.is_relative_to(frontend_root):
            raise HTTPException(status_code=404, detail="Not Found")
        if requested_path.is_file():
            return FileResponse(requested_path)
        if Path(path).suffix:
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(index_path)

    application.add_api_route(
        "/", serve_index, methods=["GET", "HEAD"], include_in_schema=False
    )
    application.add_api_route(
        "/{path:path}",
        serve_frontend_path,
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )


###############################################################################
@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = application.state.settings

    run_startup_validations(settings)
    try:
        initialize_database(settings=settings, startup=True)
    except DatabaseMigrationError as exc:
        logger.error("Database is unavailable at startup: %s", exc)
        application.state.database_available = False
        application.state.database_startup_error = (
            "The configured database could not be initialized. Use Settings > Database "
            "to test or change the connection."
        )
        application.state.job_repository = None
        application.state.job_manager = JobManager(
            max_stored_jobs=settings.jobs.max_stored_jobs
        )
        application.state.settings = settings
        yield
        return

    runtime_store = build_database_runtime_settings_store(
        settings,
        DatabaseRuntimeSettingsPersistence(get_database()),
    )
    settings = activate_runtime_settings_store(runtime_store).settings
    application.state.settings = settings
    application.state.database_available = True
    application.state.database_startup_error = None

    job_repository = JobRepository()
    job_manager = JobManager(
        max_stored_jobs=settings.jobs.max_stored_jobs,
        repository=job_repository,
    )
    job_manager.restore_persisted_jobs()
    application.state.job_repository = job_repository
    application.state.job_manager = job_manager

    yield


###############################################################################
def create_app() -> FastAPI:
    settings = get_server_settings()
    configure_logging(settings.paths.logs)

    application = FastAPI(
        title=FASTAPI_TITLE,
        version=FASTAPI_VERSION,
        description=FASTAPI_DESCRIPTION,
        lifespan=app_lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=build_cors_origins(settings),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.state.settings = settings
    application.state.database_available = True
    application.state.database_startup_error = None
    application.state.database_switch_gate = DatabaseSwitchGate()
    application.state.job_manager = JobManager(
        max_stored_jobs=settings.jobs.max_stored_jobs
    )

    @application.middleware("http")
    async def database_switch_request_gate(request, call_next):
        gate = request.app.state.database_switch_gate
        path = request.url.path.rstrip("/")
        if (
            not request.app.state.database_available
            and path != "/api/database"
            and not path.startswith("/api/database/")
            and path != "/api/health"
            and (path == "/api" or path.startswith("/api/"))
        ):
            matches_database_api_route = any(
                getattr(route, "path", "").startswith("/api/")
                and route.matches(request.scope)[0] is Match.FULL
                for route in request.app.router.routes
            )
            if matches_database_api_route:
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": request.app.state.database_startup_error
                        or "The database is unavailable."
                    },
                )
        if path == "/api/database/switch":
            return await call_next(request)
        async with gate.request():
            return await call_next(request)

    register_api_routers(application)
    register_frontend_routes(application)
    return application


app = create_app()
