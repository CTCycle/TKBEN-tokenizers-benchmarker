from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"multiprocess\.connection")

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from TKBEN.server.utils.variables import env_variables
from TKBEN.server.configurations import server_settings
from TKBEN.server.utils.constants import API_ROUTE_DOCS, API_ROUTE_ROOT
from TKBEN.server.routes.datasets import router as datasets_router
from TKBEN.server.routes.tokenizers import router as tokenizers_router
from TKBEN.server.routes.benchmarks import router as fit_router
from TKBEN.server.routes.browser import router as browser_router
from TKBEN.server.routes.jobs import router as jobs_router


###############################################################################
app = FastAPI(
    title=server_settings.fastapi.title,
    version=server_settings.fastapi.version,
    description=server_settings.fastapi.description,
)

app.include_router(datasets_router)
app.include_router(tokenizers_router)
app.include_router(fit_router)
app.include_router(browser_router)
app.include_router(jobs_router)

@app.get(API_ROUTE_ROOT)
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url=API_ROUTE_DOCS)
