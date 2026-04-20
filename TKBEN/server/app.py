from __future__ import annotations

import os
import warnings

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
from TKBEN.server.api.benchmarks import router as fit_router
from TKBEN.server.api.jobs import router as jobs_router
from TKBEN.server.api.keys import router as keys_router
from TKBEN.server.api.exports import router as exports_router

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
app = FastAPI(
    title=FASTAPI_TITLE,
    version=FASTAPI_VERSION,
    description=FASTAPI_DESCRIPTION,
)

routers = [
    datasets_router,
    tokenizers_router,
    fit_router,
    jobs_router,
    keys_router,
    exports_router,
]

for router in routers:
    app.include_router(router, prefix="/api")

if packaged_client_available():
    client_dist_path = get_client_dist_path()
    assets_path = os.path.join(client_dist_path, "assets")

    if os.path.isdir(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="spa-assets")

    @app.get("/", include_in_schema=False)
    def serve_spa_root() -> FileResponse:
        return FileResponse(os.path.join(client_dist_path, "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa_entrypoint(full_path: str) -> FileResponse:
        requested_path = os.path.join(client_dist_path, full_path)
        if os.path.isfile(requested_path):
            return FileResponse(requested_path)
        return FileResponse(os.path.join(client_dist_path, "index.html"))

else:
    @app.get("/")
    def redirect_to_docs() -> RedirectResponse:
        return RedirectResponse(url="/docs")
