from __future__ import annotations

import pytest
from datasets.exceptions import DataFilesNotFoundError
from huggingface_hub.errors import GatedRepoError
from pydantic import ValidationError
from requests import Response
from requests.exceptions import ConnectionError as RequestsConnectionError

from TKBEN.server.entities.dataset import DatasetDownloadRequest
from TKBEN.server.services.datasets import (
    HF_DATASET_ALIASES,
    DatasetService,
    LengthStatistics,
)
from TKBEN.server.services.keys import HFAccessKeyValidationError


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


def test_get_hf_access_token_for_download_returns_active_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    monkeypatch.setattr(service.key_service, "get_active_key", lambda: "hf_token")
    assert service.get_hf_access_token_for_download() == "hf_token"


def test_get_hf_access_token_for_download_falls_back_to_none_on_invalid_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()

    def raise_invalid_key() -> str:
        raise HFAccessKeyValidationError("invalid")

    monkeypatch.setattr(service.key_service, "get_active_key", raise_invalid_key)
    assert service.get_hf_access_token_for_download() is None


def test_preselected_dataset_aliases_cover_all_ui_presets() -> None:
    expected_presets = {
        "wikitext",
        "c4",
        "oscar",
        "cc_news",
        "openwebtext",
        "bookcorpus",
        "ag_news",
        "cnn_dailymail",
        "gigaword",
        "multi_news",
        "squad",
        "natural_questions",
        "hotpot_qa",
        "daily_dialog",
        "empathetic_dialogues",
        "openassistant_oasst1",
        "yelp_review_full",
        "amazon_reviews_multi",
        "imdb",
        "arxiv",
        "pubmed",
        "flores",
        "wiki40b",
        "opus_books",
    }
    assert expected_presets.issubset(set(HF_DATASET_ALIASES.keys()))


def test_the_pile_preset_is_disabled_with_clear_error() -> None:
    service = DatasetService()
    with pytest.raises(ValueError) as exc_info:
        service.resolve_dataset_download(corpus="the_pile", config=None)
    assert "disabled" in str(exc_info.value)


def configure_download_success_mocks(
    service: DatasetService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "is_dataset_in_database", lambda dataset_name: False)
    monkeypatch.setattr(service, "dataset_cached_on_disk", lambda cache_path: False)
    monkeypatch.setattr(service, "get_hf_access_token_for_download", lambda: None)
    monkeypatch.setattr(service, "find_text_column", lambda dataset: "text")
    monkeypatch.setattr(
        service,
        "collect_length_statistics",
        lambda stream_factory, **kwargs: LengthStatistics(
            document_count=2,
            total_length=11,
            min_length=5,
            max_length=6,
        ),
    )
    monkeypatch.setattr(
        service,
        "persist_dataset",
        lambda **kwargs: (
            {"bins": ["0-9"], "counts": [2], "bin_edges": [0.0, 10.0]},
            2,
        ),
    )
    monkeypatch.setattr(service, "maybe_cleanup_downloaded_source", lambda *args: None)


def test_download_and_persist_keeps_wikitext_working(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    configure_download_success_mocks(service, monkeypatch)
    captured: dict[str, str | None] = {}

    def fake_load_dataset(
        corpus: str,
        config: str | None,
        cache_dir: str | None = None,
        token: str | None = None,
        **kwargs,
    ):
        captured["corpus"] = corpus
        captured["config"] = config
        captured["cache_dir"] = cache_dir
        captured["token"] = token
        return object()

    monkeypatch.setattr("TKBEN.server.services.datasets.load_dataset", fake_load_dataset)

    result = service.download_and_persist(
        corpus="wikitext",
        config="wikitext-2-v1",
        job_id="job12345",
    )

    assert captured["corpus"] == "wikitext"
    assert captured["config"] == "wikitext-2-v1"
    assert result["dataset_name"] == "wikitext/wikitext-2-v1"


def test_download_and_persist_maps_c4_friendly_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    configure_download_success_mocks(service, monkeypatch)
    captured: dict[str, str | None] = {}

    def fake_load_dataset(
        corpus: str,
        config: str | None,
        cache_dir: str | None = None,
        token: str | None = None,
        **kwargs,
    ):
        captured["corpus"] = corpus
        captured["config"] = config
        captured["cache_dir"] = cache_dir
        captured["token"] = token
        return object()

    monkeypatch.setattr("TKBEN.server.services.datasets.load_dataset", fake_load_dataset)

    result = service.download_and_persist(
        corpus="c4",
        config=None,
        job_id="job67890",
    )

    assert captured["corpus"] == "allenai/c4"
    assert captured["config"] == "en"
    assert result["dataset_name"] == "c4/en"


def test_download_and_persist_maps_arxiv_to_canonical_hf_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    configure_download_success_mocks(service, monkeypatch)
    captured: dict[str, str | None] = {}

    def fake_load_dataset(
        corpus: str,
        config: str | None,
        cache_dir: str | None = None,
        token: str | None = None,
        **kwargs,
    ):
        captured["corpus"] = corpus
        captured["config"] = config
        return object()

    monkeypatch.setattr("TKBEN.server.services.datasets.load_dataset", fake_load_dataset)

    result = service.download_and_persist(
        corpus="arxiv",
        config=None,
        job_id="job67891",
    )

    assert captured["corpus"] == "ccdv/arxiv-summarization"
    assert captured["config"] is None
    assert result["dataset_name"] == "arxiv"


def test_download_and_persist_success_triggers_source_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    configure_download_success_mocks(service, monkeypatch)
    cleanup_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        service,
        "maybe_cleanup_downloaded_source",
        lambda cache_path, dataset_name: cleanup_calls.append((cache_path, dataset_name)),
    )

    def fake_load_dataset(
        corpus: str,
        config: str | None,
        cache_dir: str | None = None,
        token: str | None = None,
        **kwargs,
    ):
        return object()

    monkeypatch.setattr("TKBEN.server.services.datasets.load_dataset", fake_load_dataset)

    result = service.download_and_persist(
        corpus="wikitext",
        config="wikitext-2-v1",
        job_id="job-cleanup-success",
    )

    assert result["dataset_name"] == "wikitext/wikitext-2-v1"
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0][1] == "wikitext/wikitext-2-v1"


def test_download_and_persist_failed_import_does_not_cleanup_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    cleanup_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(service, "is_dataset_in_database", lambda dataset_name: False)
    monkeypatch.setattr(service, "get_hf_access_token_for_download", lambda: None)
    monkeypatch.setattr(service, "find_text_column", lambda dataset: "text")
    monkeypatch.setattr(
        service,
        "collect_length_statistics",
        lambda stream_factory, **kwargs: LengthStatistics(
            document_count=2,
            total_length=11,
            min_length=5,
            max_length=6,
        ),
    )
    monkeypatch.setattr(
        service,
        "persist_dataset",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("persist failed")),
    )
    monkeypatch.setattr(
        service,
        "maybe_cleanup_downloaded_source",
        lambda cache_path, dataset_name: cleanup_calls.append((cache_path, dataset_name)),
    )

    def fake_load_dataset(
        corpus: str,
        config: str | None,
        cache_dir: str | None = None,
        token: str | None = None,
        **kwargs,
    ):
        return object()

    monkeypatch.setattr("TKBEN.server.services.datasets.load_dataset", fake_load_dataset)

    with pytest.raises(RuntimeError, match="persist failed"):
        service.download_and_persist(
            corpus="wikitext",
            config="wikitext-2-v1",
            job_id="job-cleanup-failure",
        )

    assert cleanup_calls == []


def test_download_and_persist_uses_database_for_existence_not_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    configure_download_success_mocks(service, monkeypatch)
    database_checks: list[str] = []

    monkeypatch.setattr(
        service,
        "is_dataset_in_database",
        lambda dataset_name: database_checks.append(dataset_name) or False,
    )
    monkeypatch.setattr(
        service,
        "dataset_cached_on_disk",
        lambda cache_path: (_ for _ in ()).throw(
            AssertionError("filesystem existence checks must not be used")
        ),
    )

    def fake_load_dataset(
        corpus: str,
        config: str | None,
        cache_dir: str | None = None,
        token: str | None = None,
        **kwargs,
    ):
        return object()

    monkeypatch.setattr("TKBEN.server.services.datasets.load_dataset", fake_load_dataset)

    result = service.download_and_persist(
        corpus="wikitext",
        config="wikitext-2-v1",
        job_id="job-db-exists-only",
    )

    assert result["dataset_name"] == "wikitext/wikitext-2-v1"
    assert database_checks == ["wikitext/wikitext-2-v1"]


def test_download_and_persist_classifies_invalid_dataset_or_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    monkeypatch.setattr(service, "is_dataset_in_database", lambda dataset_name: False)
    monkeypatch.setattr(service, "dataset_cached_on_disk", lambda cache_path: False)
    monkeypatch.setattr(service, "get_hf_access_token_for_download", lambda: None)

    def raise_not_found(*args, **kwargs):
        raise DataFilesNotFoundError("No (supported) data files found.")

    monkeypatch.setattr("TKBEN.server.services.datasets.load_dataset", raise_not_found)

    with pytest.raises(RuntimeError) as exc_info:
        service.download_and_persist(corpus="unknown_dataset_name", config=None, job_id="job00001")

    message = str(exc_info.value)
    assert "invalid dataset id or configuration" in message
    assert "job=job00001" in message
    assert "'unknown_dataset_name'" in message


def test_download_and_persist_classifies_unsupported_dataset_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    monkeypatch.setattr(service, "is_dataset_in_database", lambda dataset_name: False)
    monkeypatch.setattr(service, "dataset_cached_on_disk", lambda cache_path: False)
    monkeypatch.setattr(service, "get_hf_access_token_for_download", lambda: None)

    def raise_script_error(*args, **kwargs):
        raise RuntimeError("Dataset scripts are no longer supported, but found pile.py")

    monkeypatch.setattr("TKBEN.server.services.datasets.load_dataset", raise_script_error)

    with pytest.raises(RuntimeError) as exc_info:
        service.download_and_persist(corpus="EleutherAI/pile", config="all", job_id="job00004")

    message = str(exc_info.value)
    assert "requires a legacy dataset script" in message
    assert "job=job00004" in message
    assert "'EleutherAI/pile/all'" in message


def test_download_and_persist_classifies_gated_or_auth_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    monkeypatch.setattr(service, "is_dataset_in_database", lambda dataset_name: False)
    monkeypatch.setattr(service, "dataset_cached_on_disk", lambda cache_path: False)
    monkeypatch.setattr(service, "get_hf_access_token_for_download", lambda: None)

    def raise_gated(*args, **kwargs):
        response = Response()
        response.status_code = 403
        response.url = "https://huggingface.co/datasets/oscar-corpus/oscar"
        raise GatedRepoError("Access to this dataset is restricted.", response=response)

    monkeypatch.setattr("TKBEN.server.services.datasets.load_dataset", raise_gated)

    with pytest.raises(RuntimeError) as exc_info:
        service.download_and_persist(
            corpus="oscar-corpus/OSCAR-2201",
            config="en",
            job_id="job00002",
        )

    message = str(exc_info.value)
    assert "gated or requires authentication" in message
    assert "job=job00002" in message
    assert "'oscar-corpus/OSCAR-2201/en'" in message
    assert "No valid decryptable Hugging Face token is currently configured." in message


def test_load_dataset_with_progress_reports_stage_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    progress_values: list[float] = []

    def fake_load_dataset(
        corpus: str,
        config: str | None,
        cache_dir: str | None = None,
        token: str | None = None,
        **kwargs,
    ):
        return object()

    monkeypatch.setattr("TKBEN.server.services.datasets.load_dataset", fake_load_dataset)

    service.load_dataset_with_progress(
        hf_dataset_id="wikitext",
        hf_config="wikitext-2-v1",
        cache_path="tmp",
        hf_access_token=None,
        split=None,
        progress_callback=progress_values.append,
    )

    assert progress_values
    assert progress_values[0] == 5.0
    assert progress_values[-1] == 15.0


def test_download_and_persist_classifies_network_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    monkeypatch.setattr(service, "is_dataset_in_database", lambda dataset_name: False)
    monkeypatch.setattr(service, "dataset_cached_on_disk", lambda cache_path: False)
    monkeypatch.setattr(service, "get_hf_access_token_for_download", lambda: None)

    def raise_network(*args, **kwargs):
        raise RequestsConnectionError("Connection reset by peer")

    monkeypatch.setattr("TKBEN.server.services.datasets.load_dataset", raise_network)

    with pytest.raises(RuntimeError) as exc_info:
        service.download_and_persist(corpus="wikitext", config="wikitext-2-v1", job_id="job00003")

    message = str(exc_info.value)
    assert "network/transient error" in message
    assert "job=job00003" in message
