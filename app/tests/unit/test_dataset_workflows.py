# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from collections import defaultdict
from typing import Any
import pytest
from server.common.metric_catalog import DATASET_METRIC_CATALOG
from server.services.metrics.engine import DatasetMetricsEngine
import time
from unittest.mock import MagicMock
import pandas as pd
from datasets import IterableDataset
from datasets.exceptions import DataFilesNotFoundError
from huggingface_hub.errors import GatedRepoError
from pydantic import ValidationError
from requests import Response
from requests.exceptions import ConnectionError as RequestsConnectionError
from server.contracts.dataset import DatasetDownloadRequest
from server.services.datasets import HF_DATASET_ALIASES
from server.services.datasets import DatasetService
from server.services.datasets import LengthStatistics
from server.services.keys import HFAccessKeyValidationError
from server.services.dataset_jobs import DatasetJobService
import sqlite3
from datasets import Dataset as HuggingFaceDataset
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from server.repositories.database.backend import get_database
from server.repositories.database.sqlite import SQLiteRepository
from server.repositories.schemas.models import Base
from server.repositories.schemas.models import Dataset
from server.repositories.schemas.models import DatasetDocument
from datetime import datetime
from datetime import timezone
from server.repositories.datasets import DatasetRepository

###############################################################################


DATASET_DOCUMENTS = [
    "The cat sat. The cat sat!",
    "The cat sat. The cat sat!",
    "Visit https://example.com\nEmail me@test.com\n<tag>AA 123</tag>",
    "",
]

DATASET_ENGINE_PARAMETERS: dict[str, Any] = {
    "near_empty_threshold_words": 2,
    "near_duplicate_threshold": 0.9,
    "simhash_bands": 4,
    "mattr_window": 10,
    "top_k_concentration": 20,
    "rare_tail_percent": 0.10,
    "max_vocab_in_memory": 200_000,
}

EXPECTED_DATASET_METRICS: dict[str, Any] = {
    "chars.control_ratio": 0.018018018018018018,
    "chars.digit_ratio": 0.02702702702702703,
    "chars.entropy": 4.311362346615454,
    "chars.non_ascii_ratio": 0.0,
    "chars.other_ratio": 0.6666666666666666,
    "chars.punctuation_ratio": 0.0990990990990991,
    "chars.uppercase_ratio": 0.07207207207207207,
    "chars.whitespace_ratio": 0.11711711711711711,
    "compression.avg_repetition_factor": 1.8461538461538463,
    "compression.bigram_repetition_rate": 0.19047619047619047,
    "compression.chars_per_unique_word": 8.538461538461538,
    "compression.ratio": 1.3243243243243243,
    "corpus.document_count": 4.0,
    "corpus.mattr": 0.68,
    "corpus.total_chars": 111.0,
    "corpus.total_words": 24.0,
    "corpus.ttr": 0.5416666666666666,
    "corpus.unique_words": 13.0,
    "doc.avg_word_length": [3.0, 3.0, 3.75, 0.0],
    "doc.compression_ratio": [1.44, 1.44, 1.2295081967213115, 0.0],
    "doc.length_chars": [25.0, 25.0, 61.0, 0.0],
    "doc.length_cv": 0.7834730667115998,
    "doc.length_gini": 0.41216216216216217,
    "doc.length_iqr": 25.0,
    "doc.length_kurtosis": -0.968770014018959,
    "doc.length_mad": 0.0,
    "doc.length_max": 61.0,
    "doc.length_mean": 27.75,
    "doc.length_median": 25.0,
    "doc.length_min": 0.0,
    "doc.length_p10": 0.0,
    "doc.length_p25": 0.0,
    "doc.length_p50": 25.0,
    "doc.length_p75": 25.0,
    "doc.length_p90": 61.0,
    "doc.length_p95": 61.0,
    "doc.length_p99": 61.0,
    "doc.length_skewness": 0.3733898042518787,
    "doc.std_word_length": [0.0, 0.0, 1.4215601757693317, 0.0],
    "doc.word_count": [6.0, 6.0, 12.0, 0.0],
    "hist.document_length": {
        "bin_edges": [0.0, 16.0, 32.0, 48.0, 64.0],
        "bins": ["0-15", "16-31", "32-47", "48-63"],
        "counts": [1, 2, 0, 1],
        "max_length": 61,
        "mean_length": 27.75,
        "median_length": 25.0,
        "min_length": 0,
    },
    "hist.word_length": {
        "bin_edges": [2.0, 4.0, 6.0, 8.0, 10.0],
        "bins": ["2-3", "4-5", "6-7", "8-9"],
        "counts": [19, 4, 1, 0],
        "max_length": 7,
        "mean_length": 3.375,
        "median_length": 3.0,
        "min_length": 2,
    },
    "quality.avg_sentence_count": 1.75,
    "quality.avg_sentence_length": 3.4285714285714284,
    "quality.duplicate_rate": 0.25,
    "quality.empty_rate": 0.25,
    "quality.exact_duplicate_rate": 0.25,
    "quality.is_exact_duplicate": [0.0, 1.0, 0.0, 0.0],
    "quality.is_near_duplicate": [0.0, 1.0, 0.0, 0.0],
    "quality.language_consistency": 0.5,
    "quality.language_tag": ["en", "en", "latin", "unknown"],
    "quality.near_duplicate_rate": 0.25,
    "quality.near_empty_rate": 0.25,
    "quality.sentence_length_variance": 0.5306122448979611,
    "structure.avg_paragraph_count": 0.75,
    "structure.email_density": 0.041666666666666664,
    "structure.html_tag_ratio": 0.0990990990990991,
    "structure.line_break_density": 0.018018018018018018,
    "structure.url_density": 0.041666666666666664,
    "words.dis_legomena_ratio": 0.15384615384615385,
    "words.frequency_gini": 0.32051282051282054,
    "words.hapax_ratio": 0.6153846153846154,
    "words.hhi": 0.11111111111111106,
    "words.least_common": [
        {"count": 1, "word": "123"},
        {"count": 1, "word": "aa"},
        {"count": 1, "word": "email"},
        {"count": 1, "word": "example"},
        {"count": 1, "word": "https"},
        {"count": 1, "word": "me"},
        {"count": 1, "word": "test"},
        {"count": 1, "word": "visit"},
        {"count": 2, "word": "com"},
        {"count": 2, "word": "tag"},
    ],
    "words.length_mean": 3.375,
    "words.length_median": 3.0,
    "words.length_std": 1.0728660991319774,
    "words.longest": [
        {"count": 1, "length": 7, "word": "example"},
        {"count": 1, "length": 5, "word": "email"},
        {"count": 1, "length": 5, "word": "https"},
        {"count": 1, "length": 5, "word": "visit"},
        {"count": 1, "length": 4, "word": "test"},
        {"count": 1, "length": 3, "word": "123"},
        {"count": 4, "length": 3, "word": "cat"},
        {"count": 2, "length": 3, "word": "com"},
        {"count": 4, "length": 3, "word": "sat"},
        {"count": 2, "length": 3, "word": "tag"},
        {"count": 4, "length": 3, "word": "the"},
        {"count": 1, "length": 2, "word": "aa"},
        {"count": 1, "length": 2, "word": "me"},
    ],
    "words.most_common": [
        {"count": 4, "word": "cat"},
        {"count": 4, "word": "sat"},
        {"count": 4, "word": "the"},
        {"count": 2, "word": "com"},
        {"count": 2, "word": "tag"},
        {"count": 1, "word": "123"},
        {"count": 1, "word": "aa"},
        {"count": 1, "word": "email"},
        {"count": 1, "word": "example"},
        {"count": 1, "word": "https"},
    ],
    "words.normalized_entropy": 0.923753957481481,
    "words.rare_tail_mass": 0.08333333333333333,
    "words.shannon_entropy": 3.41829583405449,
    "words.shortest": [
        {"count": 1, "length": 2, "word": "aa"},
        {"count": 1, "length": 2, "word": "me"},
        {"count": 1, "length": 3, "word": "123"},
        {"count": 4, "length": 3, "word": "cat"},
        {"count": 2, "length": 3, "word": "com"},
        {"count": 4, "length": 3, "word": "sat"},
        {"count": 2, "length": 3, "word": "tag"},
        {"count": 4, "length": 3, "word": "the"},
        {"count": 1, "length": 4, "word": "test"},
        {"count": 1, "length": 5, "word": "email"},
        {"count": 1, "length": 5, "word": "https"},
        {"count": 1, "length": 5, "word": "visit"},
        {"count": 1, "length": 7, "word": "example"},
    ],
    "words.topk_concentration": 1.0,
    "words.word_cloud": [
        {"count": 4, "weight": 100, "word": "cat"},
        {"count": 4, "weight": 100, "word": "sat"},
        {"count": 4, "weight": 100, "word": "the"},
        {"count": 2, "weight": 50, "word": "com"},
        {"count": 2, "weight": 50, "word": "tag"},
        {"count": 1, "weight": 25, "word": "123"},
        {"count": 1, "weight": 25, "word": "aa"},
        {"count": 1, "weight": 25, "word": "email"},
        {"count": 1, "weight": 25, "word": "example"},
        {"count": 1, "weight": 25, "word": "https"},
        {"count": 1, "weight": 25, "word": "me"},
        {"count": 1, "weight": 25, "word": "test"},
        {"count": 1, "weight": 25, "word": "visit"},
    ],
    "words.zipf_curve": [
        {"frequency": 4, "rank": 1, "word": "cat"},
        {"frequency": 4, "rank": 2, "word": "sat"},
        {"frequency": 4, "rank": 3, "word": "the"},
        {"frequency": 2, "rank": 4, "word": "com"},
        {"frequency": 2, "rank": 5, "word": "tag"},
        {"frequency": 1, "rank": 6, "word": "123"},
        {"frequency": 1, "rank": 7, "word": "aa"},
        {"frequency": 1, "rank": 8, "word": "email"},
        {"frequency": 1, "rank": 9, "word": "example"},
        {"frequency": 1, "rank": 10, "word": "https"},
        {"frequency": 1, "rank": 11, "word": "me"},
        {"frequency": 1, "rank": 12, "word": "test"},
        {"frequency": 1, "rank": 13, "word": "visit"},
    ],
    "words.zipf_slope": -0.7195898875112114,
}


###############################################################################
def _assert_metric_value(
    actual: Any, expected: Any, metric_key: str, path: str = ""
) -> None:
    location = f"{metric_key}{path}"
    if isinstance(expected, float):
        assert isinstance(actual, (int, float)), (
            f"{location} expected float-like numeric value, got {type(actual).__name__}: {actual}"
        )
        assert float(actual) == pytest.approx(expected, rel=1e-9, abs=1e-9), (
            f"{location} mismatch: expected {expected}, got {actual}"
        )
        return

    if isinstance(expected, int):
        assert actual == expected, (
            f"{location} mismatch: expected {expected}, got {actual}"
        )
        return

    if isinstance(expected, str):
        assert actual == expected, (
            f"{location} mismatch: expected {expected!r}, got {actual!r}"
        )
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), (
            f"{location} expected list, got {type(actual).__name__}: {actual}"
        )
        assert len(actual) == len(expected), (
            f"{location} length mismatch: expected {len(expected)}, got {len(actual)}"
        )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_metric_value(
                actual_item,
                expected_item,
                metric_key=metric_key,
                path=f"{path}[{index}]",
            )
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict), (
            f"{location} expected dict, got {type(actual).__name__}: {actual}"
        )
        assert set(actual.keys()) == set(expected.keys()), (
            f"{location} key mismatch: expected {sorted(expected.keys())}, got {sorted(actual.keys())}"
        )
        for key in sorted(expected.keys()):
            _assert_metric_value(
                actual[key],
                expected[key],
                metric_key=metric_key,
                path=f"{path}.{key}",
            )
        return

    assert actual == expected, f"{location} mismatch: expected {expected}, got {actual}"


###############################################################################
def _dataset_metric_keys() -> list[str]:
    return [
        metric["key"]
        for category in DATASET_METRIC_CATALOG
        for metric in category.get("metrics", [])
        if isinstance(metric, dict) and isinstance(metric.get("key"), str)
    ]


###############################################################################
def _extract_metric_value_map() -> dict[str, Any]:
    engine = DatasetMetricsEngine(dict(DATASET_ENGINE_PARAMETERS))
    per_document_rows: list[dict[str, Any]] = []
    for document_id, text in enumerate(DATASET_DOCUMENTS, start=1):
        per_document_rows.extend(engine.process_document(document_id, text))

    finalized = engine.finalize(histogram_bins=4)

    per_document_numeric: dict[str, dict[int, float]] = defaultdict(dict)
    per_document_text: dict[str, dict[int, str]] = defaultdict(dict)
    for row in per_document_rows:
        metric_key = row["metric_key"]
        document_id = int(row["document_id"])
        if "numeric_value" in row:
            per_document_numeric[metric_key][document_id] = float(row["numeric_value"])
        elif "text_value" in row:
            per_document_text[metric_key][document_id] = str(row["text_value"])

    aggregate_values: dict[str, Any] = {}
    for row in finalized["metric_rows"]:
        metric_key = row["metric_key"]
        if "numeric_value" in row:
            aggregate_values[metric_key] = float(row["numeric_value"])
        elif "json_value" in row:
            aggregate_values[metric_key] = row["json_value"]

    extracted: dict[str, Any] = {}
    for key in _dataset_metric_keys():
        if key == "hist.document_length":
            extracted[key] = finalized["document_histogram"]
        elif key == "hist.word_length":
            extracted[key] = finalized["word_histogram"]
        elif key in per_document_numeric:
            extracted[key] = [
                per_document_numeric[key][doc_id]
                for doc_id in sorted(per_document_numeric[key].keys())
            ]
        elif key in per_document_text:
            extracted[key] = [
                per_document_text[key][doc_id]
                for doc_id in sorted(per_document_text[key].keys())
            ]
        else:
            extracted[key] = aggregate_values[key]
    return extracted


###############################################################################
@pytest.fixture(scope="module")
def actual_dataset_metrics() -> dict[str, Any]:
    return _extract_metric_value_map()


###############################################################################
def test_dataset_metric_expected_fixture_matches_catalog() -> None:
    catalog_key_list = _dataset_metric_keys()
    catalog_keys = set(catalog_key_list)
    expected_keys = set(EXPECTED_DATASET_METRICS.keys())
    assert len(catalog_key_list) == len(catalog_keys), (
        "Dataset metric keys must be unique"
    )
    assert expected_keys == catalog_keys, (
        "Expected dataset metric fixture does not align with catalog keys. "
        f"Missing={sorted(catalog_keys - expected_keys)} "
        f"Extra={sorted(expected_keys - catalog_keys)}"
    )


###############################################################################
@pytest.mark.parametrize("metric_key", sorted(EXPECTED_DATASET_METRICS.keys()))
def test_dataset_analysis_metric_correctness(
    actual_dataset_metrics: dict[str, Any],
    metric_key: str,
) -> None:
    assert metric_key in actual_dataset_metrics, (
        f"Missing computed dataset metric '{metric_key}'"
    )
    expected = EXPECTED_DATASET_METRICS[metric_key]
    actual = actual_dataset_metrics[metric_key]
    _assert_metric_value(actual, expected, metric_key=metric_key)


###############################################################################
def test_dataset_download_request_requires_configs() -> None:
    with pytest.raises(ValidationError):
        DatasetDownloadRequest(corpus="wikitext")


###############################################################################
def test_dataset_download_request_allows_missing_configuration() -> None:
    request = DatasetDownloadRequest(corpus="c4", configs={})
    assert request.corpus == "c4"
    assert request.configs.configuration is None


###############################################################################
def test_dataset_download_request_accepts_configuration() -> None:
    request = DatasetDownloadRequest(
        corpus="wikitext",
        configs={"configuration": "wikitext-2-v1"},
    )
    assert request.configs.configuration == "wikitext-2-v1"


###############################################################################
def test_upload_existing_dataset_is_non_destructive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        service.dataset_repository,
        "delete_dataset",
        lambda dataset_name: delete_calls.append(dataset_name),
    )

    result = service.upload_and_persist(
        file_content=b"text\nhello", filename="sample.csv"
    )

    assert result == expected_payload
    assert delete_calls == []


###############################################################################
def test_get_hf_access_token_for_download_falls_back_to_none_on_invalid_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()

    def raise_invalid_key() -> str:
        raise HFAccessKeyValidationError("invalid")

    monkeypatch.setattr(service.key_service, "get_active_key", raise_invalid_key)
    assert service.get_hf_access_token_for_download() is None


###############################################################################
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


###############################################################################
def test_c4_preset_uses_bounded_streaming_training_sample() -> None:
    target = DatasetService().resolve_dataset_download(corpus="c4", config=None)

    assert target.hf_dataset_id == "allenai/c4"
    assert target.hf_config == "en"
    assert target.split == "train"
    assert target.streaming is True
    assert target.max_documents == 10000


###############################################################################
def test_the_pile_preset_is_disabled_with_clear_error() -> None:
    service = DatasetService()
    with pytest.raises(ValueError) as exc_info:
        service.resolve_dataset_download(corpus="the_pile", config=None)
    assert "disabled" in str(exc_info.value)


###############################################################################
def configure_download_success_mocks(
    service: DatasetService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        lambda **kwargs: (
            {"bins": ["0-9"], "counts": [2], "bin_edges": [0.0, 10.0]},
            2,
        ),
    )
    monkeypatch.setattr(service, "maybe_cleanup_downloaded_source", lambda *args: None)


###############################################################################
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

    monkeypatch.setattr("server.services.datasets.load_dataset", fake_load_dataset)

    result = service.download_and_persist(
        corpus="wikitext",
        config="wikitext-2-v1",
        job_id="job12345",
    )

    assert captured["corpus"] == "wikitext"
    assert captured["config"] == "wikitext-2-v1"
    assert result["dataset_name"] == "wikitext/wikitext-2-v1"


###############################################################################
def test_download_and_persist_maps_c4_friendly_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    configure_download_success_mocks(service, monkeypatch)
    captured: dict[str, object] = {}

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
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("server.services.datasets.load_dataset", fake_load_dataset)

    result = service.download_and_persist(
        corpus="c4",
        config=None,
        job_id="job67890",
    )

    assert captured["corpus"] == "allenai/c4"
    assert captured["config"] == "en"
    assert captured["kwargs"] == {"split": "train", "streaming": True}
    assert result["dataset_name"] == "c4/en"


###############################################################################
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

    monkeypatch.setattr("server.services.datasets.load_dataset", fake_load_dataset)

    result = service.download_and_persist(
        corpus="arxiv",
        config=None,
        job_id="job67891",
    )

    assert captured["corpus"] == "ccdv/arxiv-summarization"
    assert captured["config"] is None
    assert result["dataset_name"] == "arxiv"


###############################################################################
def test_download_and_persist_success_triggers_source_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    configure_download_success_mocks(service, monkeypatch)
    cleanup_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        service,
        "maybe_cleanup_downloaded_source",
        lambda cache_path, dataset_name: cleanup_calls.append(
            (cache_path, dataset_name)
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

    monkeypatch.setattr("server.services.datasets.load_dataset", fake_load_dataset)

    result = service.download_and_persist(
        corpus="wikitext",
        config="wikitext-2-v1",
        job_id="job-cleanup-success",
    )

    assert result["dataset_name"] == "wikitext/wikitext-2-v1"
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0][1] == "wikitext/wikitext-2-v1"


###############################################################################
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
        lambda cache_path, dataset_name: cleanup_calls.append(
            (cache_path, dataset_name)
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

    monkeypatch.setattr("server.services.datasets.load_dataset", fake_load_dataset)

    with pytest.raises(RuntimeError, match="persist failed"):
        service.download_and_persist(
            corpus="wikitext",
            config="wikitext-2-v1",
            job_id="job-cleanup-failure",
        )

    assert cleanup_calls == []


###############################################################################
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

    def fake_load_dataset(
        corpus: str,
        config: str | None,
        cache_dir: str | None = None,
        token: str | None = None,
        **kwargs,
    ):
        return object()

    monkeypatch.setattr("server.services.datasets.load_dataset", fake_load_dataset)

    result = service.download_and_persist(
        corpus="wikitext",
        config="wikitext-2-v1",
        job_id="job-db-exists-only",
    )

    assert result["dataset_name"] == "wikitext/wikitext-2-v1"
    assert database_checks == ["wikitext/wikitext-2-v1"]


###############################################################################
def test_download_and_persist_classifies_invalid_dataset_or_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    monkeypatch.setattr(service, "is_dataset_in_database", lambda dataset_name: False)
    monkeypatch.setattr(service, "get_hf_access_token_for_download", lambda: None)

    def raise_not_found(*args, **kwargs):
        raise DataFilesNotFoundError("No (supported) data files found.")

    monkeypatch.setattr("server.services.datasets.load_dataset", raise_not_found)

    with pytest.raises(RuntimeError) as exc_info:
        service.download_and_persist(
            corpus="unknown_dataset_name", config=None, job_id="job00001"
        )

    message = str(exc_info.value)
    assert "invalid dataset id or configuration" in message
    assert "job=job00001" in message
    assert "'unknown_dataset_name'" in message


###############################################################################
def test_download_and_persist_classifies_unsupported_dataset_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    monkeypatch.setattr(service, "is_dataset_in_database", lambda dataset_name: False)
    monkeypatch.setattr(service, "get_hf_access_token_for_download", lambda: None)

    def raise_script_error(*args, **kwargs):
        raise RuntimeError("Dataset scripts are no longer supported, but found pile.py")

    monkeypatch.setattr("server.services.datasets.load_dataset", raise_script_error)

    with pytest.raises(RuntimeError) as exc_info:
        service.download_and_persist(
            corpus="EleutherAI/pile", config="all", job_id="job00004"
        )

    message = str(exc_info.value)
    assert "requires a legacy dataset script" in message
    assert "job=job00004" in message
    assert "'EleutherAI/pile/all'" in message


###############################################################################
def test_download_and_persist_classifies_gated_or_auth_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    monkeypatch.setattr(service, "is_dataset_in_database", lambda dataset_name: False)
    monkeypatch.setattr(service, "get_hf_access_token_for_download", lambda: None)

    def raise_gated(*args, **kwargs):
        response = Response()
        response.status_code = 403
        response.url = "https://huggingface.co/datasets/oscar-corpus/oscar"
        raise GatedRepoError("Access to this dataset is restricted.", response=response)

    monkeypatch.setattr("server.services.datasets.load_dataset", raise_gated)

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


###############################################################################
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

    monkeypatch.setattr("server.services.datasets.load_dataset", fake_load_dataset)

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


###############################################################################
def test_load_dataset_with_progress_limits_streaming_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()

    streaming_dataset = MagicMock(spec=IterableDataset)
    limited_streaming_dataset = object()
    streaming_dataset.take.return_value = limited_streaming_dataset
    captured: dict[str, object] = {}

    def fake_load_dataset(*args, **kwargs):
        captured.update(kwargs)
        return streaming_dataset

    monkeypatch.setattr("server.services.datasets.load_dataset", fake_load_dataset)

    limited_dataset = service.load_dataset_with_progress(
        hf_dataset_id="allenai/c4",
        hf_config="en",
        cache_path="tmp",
        hf_access_token=None,
        split="train",
        streaming=True,
        max_documents=2,
    )

    assert captured["split"] == "train"
    assert captured["streaming"] is True
    streaming_dataset.take.assert_called_once_with(2)
    assert limited_dataset is limited_streaming_dataset


###############################################################################
def test_download_and_persist_stops_before_retry_when_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    configure_download_success_mocks(service, monkeypatch)
    service.download_retry_attempts = 3
    state = {"attempts": 0}
    cleanup_calls: list[str] = []

    def raise_network(*args, **kwargs):
        state["attempts"] += 1
        raise RequestsConnectionError("Connection reset by peer")

    monkeypatch.setattr(service, "load_dataset_with_progress", raise_network)
    monkeypatch.setattr(
        service,
        "cleanup_cancelled_dataset",
        lambda dataset_name: cleanup_calls.append(dataset_name),
    )

    result = service.download_and_persist(
        corpus="wikitext",
        config="wikitext-2-v1",
        should_stop=lambda: state["attempts"] > 0,
        job_id="job-cancelled-download",
    )

    assert result == {}
    assert state["attempts"] == 1
    assert cleanup_calls == ["wikitext/wikitext-2-v1"]


###############################################################################
def test_download_and_persist_classifies_network_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    monkeypatch.setattr(service, "is_dataset_in_database", lambda dataset_name: False)
    monkeypatch.setattr(service, "get_hf_access_token_for_download", lambda: None)

    def raise_network(*args, **kwargs):
        raise RequestsConnectionError("Connection reset by peer")

    monkeypatch.setattr("server.services.datasets.load_dataset", raise_network)

    with pytest.raises(RuntimeError) as exc_info:
        service.download_and_persist(
            corpus="wikitext", config="wikitext-2-v1", job_id="job00003"
        )

    message = str(exc_info.value)
    assert "network/transient error" in message
    assert "job=job00003" in message


###############################################################################
def test_download_and_persist_retries_transient_failures_with_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    configure_download_success_mocks(service, monkeypatch)
    service.download_retry_attempts = 3
    service.download_retry_backoff_seconds = 0.25
    sleep_calls: list[float] = []
    attempts = {"count": 0}

    def fake_load_dataset(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RequestsConnectionError("Connection reset by peer")
        return object()

    monkeypatch.setattr("server.services.datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        "server.services.dataset_operations.time.sleep", sleep_calls.append
    )

    result = service.download_and_persist(
        corpus="wikitext",
        config="wikitext-2-v1",
        job_id="job-retry-success",
    )

    assert result["dataset_name"] == "wikitext/wikitext-2-v1"
    assert attempts["count"] == 3
    assert sleep_calls == [0.25, 0.5]


###############################################################################
def test_download_and_persist_does_not_retry_non_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    monkeypatch.setattr(service, "is_dataset_in_database", lambda dataset_name: False)
    monkeypatch.setattr(service, "get_hf_access_token_for_download", lambda: None)
    service.download_retry_attempts = 4
    sleep_calls: list[float] = []
    attempts = {"count": 0}

    def raise_not_found(*args, **kwargs):
        attempts["count"] += 1
        raise DataFilesNotFoundError("No (supported) data files found.")

    monkeypatch.setattr("server.services.datasets.load_dataset", raise_not_found)
    monkeypatch.setattr(
        "server.services.dataset_operations.time.sleep", sleep_calls.append
    )

    with pytest.raises(RuntimeError) as exc_info:
        service.download_and_persist(
            corpus="unknown_dataset_name",
            config=None,
            job_id="job-no-retry-invalid",
        )

    assert "invalid dataset id or configuration" in str(exc_info.value)
    assert attempts["count"] == 1
    assert sleep_calls == []


###############################################################################
def test_download_and_persist_timeout_is_reported_as_provider_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    configure_download_success_mocks(service, monkeypatch)
    service.download_timeout_seconds = 0.01
    service.download_retry_attempts = 1

    def slow_load_dataset(*args, **kwargs):
        time.sleep(0.05)
        return object()

    monkeypatch.setattr("server.services.datasets.load_dataset", slow_load_dataset)

    with pytest.raises(RuntimeError) as exc_info:
        service.download_and_persist(
            corpus="wikitext",
            config="wikitext-2-v1",
            job_id="job-timeout",
        )

    message = str(exc_info.value)
    assert "did not finish loading" in message
    assert "job=job-timeout" in message


###############################################################################
def test_upload_and_persist_hides_internal_parser_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetService()
    monkeypatch.setattr(service, "is_dataset_in_database", lambda dataset_name: False)

    def raise_parser_error(*args, **kwargs):
        raise pd.errors.ParserError("internal parser stack details")

    monkeypatch.setattr("server.services.datasets.pd.read_csv", raise_parser_error)

    with pytest.raises(ValueError) as exc_info:
        service.upload_and_persist(file_content=b"text\nx", filename="sample.csv")

    assert str(exc_info.value) == (
        "Could not read the uploaded file. Check the file format and try again."
    )


###############################################################################
def test_build_analysis_payload_preserves_contract() -> None:
    service = DatasetJobService()
    payload = service.build_analysis_payload(
        {
            "dataset_name": "custom/demo",
            "report_id": 1,
            "report_version": 2,
            "document_length_histogram": {"bins": ["1-2"], "counts": [1]},
            "word_length_histogram": {"bins": ["1-2"], "counts": [1]},
        }
    )
    assert payload["status"] == "success"
    assert payload["dataset_name"] == "custom/demo"
    assert payload["report_id"] == 1


###############################################################################
def test_extract_configuration_handles_missing() -> None:
    service = DatasetJobService()
    assert service.extract_configuration({}) is None
    assert (
        service.extract_configuration({"configs": {"configuration": " abc "}}) == "abc"
    )


###############################################################################
def _build_service(monkeypatch: pytest.MonkeyPatch) -> tuple[DatasetService, Engine]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    event.listen(engine, "connect", SQLiteRepository.enable_foreign_keys)
    Base.metadata.create_all(engine, checkfirst=True)
    backend = get_database().backend
    monkeypatch.setattr(backend, "engine", engine)
    monkeypatch.setattr(
        backend,
        "session_factory",
        sessionmaker(bind=engine, future=True),
    )
    service = DatasetService()
    service.streaming_batch_size = 1
    service.log_interval = 1
    return service, engine


###############################################################################
def test_near_limit_csv_import_persists_large_text(monkeypatch) -> None:
    service, engine = _build_service(monkeypatch)
    try:
        limit = service.settings.max_upload_bytes
        near_limit_size = limit - 1024 * 1024
        header = b"text\n"
        file_content = header + b"x" * (near_limit_size - len(header))

        result = service.upload_and_persist(
            file_content=file_content,
            filename="near-limit.csv",
        )

        assert result["dataset_name"] == "custom/near-limit"
        assert result["document_count"] == 1
        assert result["saved_count"] == 1
        with Session(bind=engine) as session:
            dataset = session.scalar(
                select(Dataset).where(Dataset.name == "custom/near-limit")
            )
            assert dataset is not None
            text_length = session.scalar(
                select(func.length(DatasetDocument.text)).where(
                    DatasetDocument.dataset_id == dataset.id
                )
            )

        assert dataset.status == "ready"
        assert text_length == near_limit_size - len(header)
    finally:
        engine.dispose()


###############################################################################
@pytest.mark.parametrize("import_kind", ["upload", "download"])
def test_sqlite_full_cleans_partial_import_and_retry_succeeds(
    monkeypatch,
    import_kind: str,
) -> None:
    service, engine = _build_service(monkeypatch)
    original_save_batch = service.dataset_repository.save_document_batch
    saved_batches = 0
    dataset_name = (
        "custom/recoverable" if import_kind == "upload" else "public/recoverable"
    )

    def save_batch_and_exhaust_database(batch: list[dict[str, Any]]) -> None:
        nonlocal saved_batches
        original_save_batch(batch)
        saved_batches += 1
        if saved_batches == 1:
            with engine.connect() as connection:
                page_count = connection.exec_driver_sql(
                    "PRAGMA page_count"
                ).scalar_one()
                connection.exec_driver_sql(f"PRAGMA max_page_count={page_count}")
                connection.commit()

    monkeypatch.setattr(
        service.dataset_repository,
        "save_document_batch",
        save_batch_and_exhaust_database,
    )

    def persist_import() -> int:
        if import_kind == "upload":
            result = service.upload_and_persist(
                file_content=b"text\nfirst\n" + b"x" * 100_000,
                filename="recoverable.csv",
            )
            return result["saved_count"]

        dataset = HuggingFaceDataset.from_dict({"text": ["first", "x" * 100_000]})
        stats = service.collect_length_statistics(
            service.dataset_length_stream(dataset, "text", remove_invalid=True)
        )
        _, saved_count = service.persist_dataset(
            dataset=dataset,
            dataset_name=dataset_name,
            text_column="text",
            stats=stats,
            remove_invalid=True,
        )
        return saved_count

    try:
        with pytest.raises(OperationalError) as error:
            persist_import()

        assert error.value.orig.sqlite_errorcode == sqlite3.SQLITE_FULL
        with Session(bind=engine) as session:
            leftover = session.scalar(
                select(Dataset).where(Dataset.name == dataset_name)
            )
            leftover_documents = session.scalar(select(func.count(DatasetDocument.id)))
        assert leftover is None
        assert leftover_documents == 0

        with engine.connect() as connection:
            connection.exec_driver_sql("PRAGMA max_page_count=2147483646")
            connection.commit()
        monkeypatch.setattr(
            service.dataset_repository,
            "save_document_batch",
            original_save_batch,
        )

        assert persist_import() == 2
        with Session(bind=engine) as session:
            dataset = session.scalar(
                select(Dataset).where(Dataset.name == dataset_name)
            )
            assert dataset is not None
            document_count = session.scalar(
                select(func.count(DatasetDocument.id)).where(
                    DatasetDocument.dataset_id == dataset.id
                )
            )

        assert dataset.status == "ready"
        assert document_count == 2
    finally:
        engine.dispose()


###############################################################################
def test_streaming_preserves_empty_and_unicode_rows(monkeypatch) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)

    with Session(bind=engine) as session:
        now = datetime.now(timezone.utc)
        dataset = Dataset(
            name="custom/stream",
            status="ready",
            created_at=now,
            updated_at=now,
            ready_at=now,
        )
        session.add(dataset)
        session.flush()
        session.add_all(
            [
                DatasetDocument(dataset_id=dataset.id, ordinal=0, text=""),
                DatasetDocument(dataset_id=dataset.id, ordinal=1, text=" "),
                DatasetDocument(dataset_id=dataset.id, ordinal=2, text="emoji 😀"),
                DatasetDocument(dataset_id=dataset.id, ordinal=3, text="CJK 漢字"),
            ]
        )
        session.commit()

    repository = DatasetRepository()
    rows = list(
        repository.iterate_dataset_rows_for_benchmarks("custom/stream", batch_size=2)
    )
    assert [text for _, text in rows] == ["", " ", "emoji 😀", "CJK 漢字"]


###############################################################################
def test_validation_preview_reports_filtered_and_sampled_counts(monkeypatch) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)

    with Session(bind=engine) as session:
        now = datetime.now(timezone.utc)
        dataset = Dataset(
            name="custom/preview",
            status="ready",
            created_at=now,
            updated_at=now,
            ready_at=now,
        )
        session.add(dataset)
        session.flush()
        session.add_all(
            DatasetDocument(dataset_id=dataset.id, ordinal=index, text=text)
            for index, text in enumerate(["", " ", "abcde", "abcdefghij"])
        )
        session.commit()

    service = DatasetService()
    unfiltered = service.preview_validation("custom/preview")
    assert unfiltered["total_documents"] == 4
    assert unfiltered["empty_documents"] == 2
    assert (unfiltered["min_length"], unfiltered["max_length"]) == (0, 10)
    assert unfiltered["matching_documents"] == 4
    assert sum(bin_["count"] for bin_ in unfiltered["length_histogram"]) == 4

    non_empty = service.preview_validation(
        "custom/preview", sampling={"fraction": 0.5}, filters={"exclude_empty": True}
    )
    assert non_empty["matching_documents"] == 2
    assert non_empty["sampled_documents"] == 1

    long_only = service.preview_validation(
        "custom/preview", sampling={"count": 1000}, filters={"min_length": 6}
    )
    assert long_only["matching_documents"] == 1
    assert long_only["sampled_documents"] == 1


###############################################################################
def test_large_dataset_streaming_batches_do_not_materialize_all_rows(
    monkeypatch,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)
    with Session(bind=engine) as session:
        now = datetime.now(timezone.utc)
        dataset = Dataset(
            name="custom/large_stream",
            status="ready",
            created_at=now,
            updated_at=now,
            ready_at=now,
        )
        session.add(dataset)
        session.flush()
        session.add_all(
            [
                DatasetDocument(
                    dataset_id=dataset.id, ordinal=index, text=f"row-{index}"
                )
                for index in range(2000)
            ]
        )
        session.commit()

    repository = DatasetRepository()
    batches = list(
        repository.iterate_dataset_batches("custom/large_stream", batch_size=128)
    )
    assert len(batches) >= 10
    assert sum(len(batch) for batch in batches) == 2000
