from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from server.api.benchmarks import router as benchmarks_router
from server.api.datasets import router as datasets_router
from server.api.exports import router as exports_router
from server.api.jobs import router as jobs_router
from server.api.keys import router as keys_router
from server.api.tokenizers import router as tokenizers_router
from server.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)

from server.contracts.health import HealthResponse
from server.configurations import get_server_settings
from server.repositories.database.initializer import initialize_database
from server.services.jobs import JobManager
from server.services.startup_validation import (
    build_cors_origins,
    run_startup_validations,
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
        exports_router,
    ):
        application.include_router(router, prefix="/api")

###############################################################################
def register_frontend_routes(application: FastAPI) -> None:
    application.add_api_route("/", redirect_to_docs, methods=["GET"])

###############################################################################
@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = application.state.settings

    run_startup_validations()
    initialize_database(settings=settings, startup=True)

    yield

###############################################################################
def create_app() -> FastAPI:
    application = FastAPI(
        title=FASTAPI_TITLE,
        version=FASTAPI_VERSION,
        description=FASTAPI_DESCRIPTION,
        lifespan=app_lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=build_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    settings = get_server_settings()
    application.state.settings = settings
    terminal_retention_seconds = settings.jobs.terminal_retention_seconds
    application.state.job_manager = JobManager(
        terminal_retention_seconds=terminal_retention_seconds
    )
    register_api_routers(application)
    register_frontend_routes(application)
    return application


app = create_app()
