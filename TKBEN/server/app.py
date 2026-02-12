from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"multiprocess\.connection")

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from TKBEN.server.common.constants import (
    API_ROUTE_DOCS,
    API_ROUTE_ROOT,
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from TKBEN.server.routes.datasets import router as datasets_router
from TKBEN.server.routes.tokenizers import router as tokenizers_router
from TKBEN.server.routes.benchmarks import router as fit_router
from TKBEN.server.routes.jobs import router as jobs_router
from TKBEN.server.routes.keys import router as keys_router


###############################################################################
app = FastAPI(
    title=FASTAPI_TITLE,
    version=FASTAPI_VERSION,
    description=FASTAPI_DESCRIPTION,
)

app.include_router(datasets_router)
app.include_router(tokenizers_router)
app.include_router(fit_router)
app.include_router(jobs_router)
app.include_router(keys_router)

@app.get(API_ROUTE_ROOT)
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url=API_ROUTE_DOCS)

