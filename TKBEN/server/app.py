from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from TKBEN.server.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from TKBEN.server.api.datasets import router as datasets_router
from TKBEN.server.api.tokenizers import router as tokenizers_router
from TKBEN.server.api.benchmarks import router as benchmarks_router
from TKBEN.server.api.jobs import router as jobs_router
from TKBEN.server.api.keys import router as keys_router
from TKBEN.server.api.exports import router as exports_router
from TKBEN.server.services.jobs import JobManager

###############################################################################
def tauri_mode_enabled() -> bool:
    value = os.getenv("TKBEN_TAURI_MODE", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}

###############################################################################
def get_client_dist_path() -> str:
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_path, "client", "dist")

###############################################################################
def packaged_client_available() -> bool:
    return tauri_mode_enabled() and os.path.isdir(get_client_dist_path())


###############################################################################
def serve_spa_root() -> FileResponse:
    client_dist_path = get_client_dist_path()
    return FileResponse(os.path.join(client_dist_path, "index.html"))


###############################################################################
def serve_spa_entrypoint(full_path: str) -> FileResponse:
    client_dist_path = get_client_dist_path()
    requested_path = os.path.join(client_dist_path, full_path)
    if os.path.isfile(requested_path):
        return FileResponse(requested_path)
    return FileResponse(os.path.join(client_dist_path, "index.html"))


###############################################################################
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


###############################################################################
def register_api_routers(application: FastAPI) -> None:
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
    if not packaged_client_available():
        application.add_api_route("/", redirect_to_docs, methods=["GET"])
        return

    client_dist_path = get_client_dist_path()
    assets_path = os.path.join(client_dist_path, "assets")

    if os.path.isdir(assets_path):
        application.mount(
            "/assets",
            StaticFiles(directory=assets_path),
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


###############################################################################
def create_app() -> FastAPI:
    application = FastAPI(
        title=FASTAPI_TITLE,
        version=FASTAPI_VERSION,
        description=FASTAPI_DESCRIPTION,
    )
    application.state.job_manager = JobManager()
    register_api_routers(application)
    register_frontend_routes(application)
    return application


###############################################################################
app = create_app()
