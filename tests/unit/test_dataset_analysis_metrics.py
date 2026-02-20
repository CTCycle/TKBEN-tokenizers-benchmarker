from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest

from TKBEN.server.services.metrics.catalog import DATASET_METRIC_CATALOG
from TKBEN.server.services.metrics.engine import DatasetMetricsEngine


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
    "chars.other_ratio": 0.6486486486486486,
    "chars.punctuation_ratio": 0.0990990990990991,
    "chars.uppercase_ratio": 0.07207207207207207,
    "chars.whitespace_ratio": 0.13513513513513514,
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


def _assert_metric_value(actual: Any, expected: Any, metric_key: str, path: str = "") -> None:
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
        assert actual == expected, f"{location} mismatch: expected {expected}, got {actual}"
        return

    if isinstance(expected, str):
        assert actual == expected, f"{location} mismatch: expected {expected!r}, got {actual!r}"
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


def _dataset_metric_keys() -> list[str]:
    return [
        metric["key"]
        for category in DATASET_METRIC_CATALOG
        for metric in category.get("metrics", [])
        if isinstance(metric, dict) and isinstance(metric.get("key"), str)
    ]


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


@pytest.fixture(scope="module")
def actual_dataset_metrics() -> dict[str, Any]:
    return _extract_metric_value_map()


def test_dataset_metric_catalog_has_expected_cardinality() -> None:
    keys = _dataset_metric_keys()
    assert len(keys) == 77, f"Expected 77 dataset metrics, found {len(keys)}"
    assert len(set(keys)) == 77, "Dataset metric keys must be unique"


def test_dataset_metric_expected_fixture_matches_catalog() -> None:
    catalog_keys = set(_dataset_metric_keys())
    expected_keys = set(EXPECTED_DATASET_METRICS.keys())
    assert expected_keys == catalog_keys, (
        "Expected dataset metric fixture does not align with catalog keys. "
        f"Missing={sorted(catalog_keys - expected_keys)} "
        f"Extra={sorted(expected_keys - catalog_keys)}"
    )


@pytest.mark.parametrize("metric_key", sorted(EXPECTED_DATASET_METRICS.keys()))
def test_dataset_analysis_metric_correctness(
    actual_dataset_metrics: dict[str, Any],
    metric_key: str,
) -> None:
    assert metric_key in actual_dataset_metrics, f"Missing computed dataset metric '{metric_key}'"
    expected = EXPECTED_DATASET_METRICS[metric_key]
    actual = actual_dataset_metrics[metric_key]
    _assert_metric_value(actual, expected, metric_key=metric_key)
