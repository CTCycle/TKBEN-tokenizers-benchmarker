from __future__ import annotations

import pytest
from pydantic import ValidationError

from TKBEN.server.schemas.dataset import DatasetDownloadRequest
from TKBEN.server.services.datasets import DatasetService


def test_dataset_download_request_requires_configs() -> None:
    with pytest.raises(ValidationError):
        DatasetDownloadRequest(corpus="wikitext")


def test_dataset_download_request_allows_missing_configuration() -> None:
    request = DatasetDownloadRequest(corpus="c4", configs={})
    assert request.corpus == "c4"
    assert request.configs.configuration is None


def test_dataset_download_request_accepts_configuration() -> None:
    request = DatasetDownloadRequest(
        corpus="wikitext",
        configs={"configuration": "wikitext-2-v1"},
    )
    assert request.configs.configuration == "wikitext-2-v1"


def test_pickler_compatibility_patch_allows_datasets_dill_dump() -> None:
    DatasetService._hf_pickler_patch_applied = False
    DatasetService.ensure_datasets_pickler_compatibility()

    from datasets.utils import _dill as datasets_dill

    payload = datasets_dill.dumps({"a": 1, "b": 2})
    assert isinstance(payload, bytes)
    assert payload
