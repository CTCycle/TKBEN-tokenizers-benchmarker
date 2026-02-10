from __future__ import annotations

import pytest
from pydantic import ValidationError

from TKBEN.server.entities.dataset import DatasetDownloadRequest
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


def test_datasets_dill_dump_works_without_compatibility_patch() -> None:
    from datasets.utils import _dill as datasets_dill

    payload = datasets_dill.dumps({"a": 1, "b": 2})
    assert isinstance(payload, bytes)
    assert payload


def test_upload_existing_dataset_is_non_destructive(monkeypatch: pytest.MonkeyPatch) -> None:
    service = DatasetService()
    expected_payload = {
        "dataset_name": "custom/sample",
        "text_column": "text",
        "document_count": 3,
        "saved_count": 3,
        "histogram": {"bins": [], "counts": [], "bin_edges": []},
    }
    delete_calls: list[str] = []

    monkeypatch.setattr(
        service,
        "is_dataset_in_database",
        lambda dataset_name: dataset_name == "custom/sample",
    )
    monkeypatch.setattr(
        service,
        "build_persisted_dataset_payload",
        lambda dataset_name, text_column="text": expected_payload,
    )
    monkeypatch.setattr(
        service.dataset_serializer,
        "delete_dataset",
        lambda dataset_name: delete_calls.append(dataset_name),
    )

    result = service.upload_and_persist(file_content=b"text\nhello", filename="sample.csv")

    assert result == expected_payload
    assert delete_calls == []
