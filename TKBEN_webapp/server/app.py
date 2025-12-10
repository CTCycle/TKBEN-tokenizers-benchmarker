from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from TKBEN_webapp.server.utils.variables import env_variables
from TKBEN_webapp.server.utils.configurations import server_settings
from TKBEN_webapp.server.routes.tokenizers import router as tokenizers_router
from TKBEN_webapp.server.routes.benchmarks import router as fit_router


###############################################################################
app = FastAPI(
    title=server_settings.fastapi.title,
    version=server_settings.fastapi.version,
    description=server_settings.fastapi.description,
)

app.include_router(tokenizers_router)
app.include_router(fit_router)

@app.get("/")
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")
