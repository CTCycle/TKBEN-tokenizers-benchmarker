from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from TKBEN_webapp.server.schemas.benchmarks import DatasetLoadResponse
from TKBEN_webapp.server.utils.logger import logger
from TKBEN_webapp.server.utils.services.datasets import DatasetService

router = APIRouter(prefix="/datasets", tags=["load"])
dataset_service = DatasetService()


