from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

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
from server.common.path import (
    CLIENT_ASSETS_PATH,
    CLIENT_DIST_PATH,
    CLIENT_INDEX_FILE_PATH,
)
from server.configurations import get_server_settings
from server.repositories.database.initializer import initialize_database
from server.services.jobs import JobManager
from server.services.startup_validation import (
    build_cors_origins,
    run_startup_validations,
)


def tauri_mode_enabled() -> bool:
    value = os.getenv("TKBEN_TAURI_MODE", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def packaged_client_available() -> bool:
    return CLIENT_INDEX_FILE_PATH.is_file()


def serve_spa_root() -> FileResponse:
    return FileResponse(CLIENT_INDEX_FILE_PATH)


def serve_spa_entrypoint(full_path: str) -> FileResponse:
    client_root = CLIENT_DIST_PATH.resolve()
    requested_path = (client_root / full_path).resolve()

    if requested_path.is_relative_to(client_root) and requested_path.is_file():
        return FileResponse(requested_path)

    return FileResponse(CLIENT_INDEX_FILE_PATH)


def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


def backend_healthcheck() -> dict[str, str]:
    return {"status": "ok"}


def register_api_routers(application: FastAPI) -> None:
    application.add_api_route("/api/health", backend_healthcheck, methods=["GET"])
    for router in (
        datasets_router,
        tokenizers_router,
        benchmarks_router,
        jobs_router,
        keys_router,
        exports_router,
    ):
        application.include_router(router, prefix="/api")


def register_frontend_routes(application: FastAPI) -> None:
    if not tauri_mode_enabled() or not packaged_client_available():
        application.add_api_route("/", redirect_to_docs, methods=["GET"])
        return

    if CLIENT_ASSETS_PATH.is_dir():
        application.mount(
            "/assets",
            StaticFiles(directory=CLIENT_ASSETS_PATH),
            name="spa-assets",
        )

    application.add_api_route(
        "/",
        serve_spa_root,
        methods=["GET"],
        include_in_schema=False,
    )
    application.add_api_route(
        "/{full_path:path}",
        serve_spa_entrypoint,
        methods=["GET"],
        include_in_schema=False,
    )


@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_server_settings()

    run_startup_validations(
        tauri_mode_enabled=tauri_mode_enabled(),
        client_index_file_path=CLIENT_INDEX_FILE_PATH,
    )
    initialize_database()

    application.state.settings = settings
    yield


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
    application.state.job_manager = JobManager()
    register_api_routers(application)
    register_frontend_routes(application)
    return application


app = create_app()
