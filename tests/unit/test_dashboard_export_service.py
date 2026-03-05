from __future__ import annotations

import pytest

from TKBEN.server.services.dashboard_export import DashboardExportService


def build_dataset_payload() -> dict:
    return {
        "report": {
            "dataset_name": "wikitext/wikitext-2-v1",
            "report_id": 12,
            "created_at": "2026-03-04T12:00:00Z",
            "document_count": 5000,
            "document_length_histogram": {
                "bins": ["0-10", "10-20", "20-30", "30-40"],
                "counts": [12, 84, 66, 18],
            },
            "word_length_histogram": {
                "bins": ["1", "2", "3", "4", "5+"],
                "counts": [25, 47, 61, 39, 22],
            },
            "most_common_words": [
                {"word": "the", "count": 250},
                {"word": "of", "count": 210},
                {"word": "and", "count": 196},
                {"word": "to", "count": 182},
            ],
            "aggregate_statistics": {
                "doc.length_mean": 74.82,
                "doc.length_min": 1,
                "doc.length_max": 510,
                "doc.length_cv": 0.43,
                "doc.length_p50": 68,
                "doc.length_p90": 132,
                "doc.length_p99": 228,
                "corpus.unique_words": 15230,
                "corpus.mattr": 0.8312,
                "words.shannon_entropy": 7.2345,
                "words.hapax_ratio": 0.5123,
                "words.zipf_slope": -1.0211,
                "words.frequency_gini": 0.7563,
                "words.hhi": 0.001421,
                "chars.whitespace_ratio": 0.21,
                "chars.punctuation_ratio": 0.09,
                "chars.digit_ratio": 0.02,
                "chars.uppercase_ratio": 0.05,
                "chars.non_ascii_ratio": 0.01,
                "chars.control_ratio": 0.0,
                "chars.other_ratio": 0.62,
                "quality.duplicate_rate": 0.031,
                "quality.near_duplicate_rate": 0.072,
                "words.topk_concentration": 0.412,
                "words.rare_tail_mass": 0.133,
                "words.normalized_entropy": 0.74,
                "words.zipf_curve": [
                    [1, 300],
                    [2, 205],
                    [3, 150],
                    [4, 110],
                ],
            },
        }
    }


def build_tokenizer_payload() -> dict:
    return {
        "report": {
            "report_id": 22,
            "tokenizer_name": "bert-base-uncased",
            "vocabulary_size": 30522,
            "huggingface_url": "https://huggingface.co/bert-base-uncased",
            "global_stats": {
                "tokenizer_class": "BertTokenizerFast",
                "base_vocabulary_size": 30522,
                "model_max_length": 512,
                "padding_side": "right",
                "special_tokens_count": 5,
                "added_tokens_count": 0,
            },
            "token_length_histogram": {
                "bins": ["1", "2", "3", "4", "5", "6+"],
                "counts": [1220, 8045, 9102, 6703, 3987, 1465],
            },
        },
        "vocabulary_items": [
            {"token_id": 0, "token": "[PAD]", "length": 5},
            {"token_id": 1, "token": "[UNK]", "length": 5},
            {"token_id": 2, "token": "[CLS]", "length": 5},
            {"token_id": 3, "token": "[SEP]", "length": 5},
            {"token_id": 4, "token": "[MASK]", "length": 6},
            {"token_id": 5, "token": "the", "length": 3},
        ],
    }


def build_benchmark_payload() -> dict:
    return {
        "report": {
            "report_id": 8,
            "dataset_name": "wikitext/wikitext-2-v1",
            "documents_processed": 3000,
            "tokenizers_count": 3,
            "global_metrics": [
                {
                    "tokenizer": "bert-base-uncased",
                    "oov_rate": 0.012,
                    "round_trip_fidelity_rate": 0.98,
                    "word_recovery_rate": 0.95,
                    "character_coverage": 0.99,
                    "subword_fertility": 1.32,
                    "token_distribution_entropy": 7.01,
                },
                {
                    "tokenizer": "gpt2",
                    "oov_rate": 0.018,
                    "round_trip_fidelity_rate": 0.96,
                    "word_recovery_rate": 0.92,
                    "character_coverage": 0.98,
                    "subword_fertility": 1.41,
                    "token_distribution_entropy": 6.71,
                },
            ],
            "chart_data": {
                "speed_metrics": [
                    {"tokenizer": "bert-base-uncased", "tokens_per_second": 11000},
                    {"tokenizer": "gpt2", "tokens_per_second": 9600},
                ],
                "vocabulary_stats": [
                    {"tokenizer": "bert-base-uncased", "vocabulary_size": 30522},
                    {"tokenizer": "gpt2", "vocabulary_size": 50257},
                ],
                "token_length_distributions": [
                    {
                        "tokenizer": "bert-base-uncased",
                        "bins": [
                            {"bin_start": 1, "bin_end": 2, "count": 230},
                            {"bin_start": 3, "bin_end": 4, "count": 410},
                        ],
                    }
                ],
            },
        },
        "selected_distribution_tokenizer": "bert-base-uncased",
    }


def test_export_dataset_dashboard_pdf_generates_pdf_bytes() -> None:
    service = DashboardExportService()
    result = service.export_dashboard_pdf(
        dashboard_type="dataset",
        report_name="dataset-layout-export",
        file_name="dataset-layout-export",
        dashboard_payload=build_dataset_payload(),
    )

    assert result.file_name == "dataset-layout-export.pdf"
    assert result.page_count >= 2
    assert result.pdf_bytes.startswith(b"%PDF")
    assert len(result.pdf_bytes) > 5000


def test_export_tokenizer_dashboard_pdf_generates_pdf_bytes() -> None:
    service = DashboardExportService()
    result = service.export_dashboard_pdf(
        dashboard_type="tokenizer",
        report_name="tokenizer-layout-export",
        file_name="tokenizer-layout-export",
        dashboard_payload=build_tokenizer_payload(),
    )

    assert result.file_name == "tokenizer-layout-export.pdf"
    assert result.page_count >= 1
    assert result.pdf_bytes.startswith(b"%PDF")
    assert len(result.pdf_bytes) > 4000


def test_export_benchmark_dashboard_pdf_generates_pdf_bytes() -> None:
    service = DashboardExportService()
    result = service.export_dashboard_pdf(
        dashboard_type="benchmark",
        report_name="benchmark-layout-export",
        file_name="benchmark-layout-export",
        dashboard_payload=build_benchmark_payload(),
    )

    assert result.file_name == "benchmark-layout-export.pdf"
    assert result.page_count >= 2
    assert result.pdf_bytes.startswith(b"%PDF")
    assert len(result.pdf_bytes) > 5000


def test_export_dashboard_pdf_rejects_unsupported_dashboard_type() -> None:
    service = DashboardExportService()
    with pytest.raises(ValueError, match="Unsupported dashboard type"):
        service.export_dashboard_pdf(
            dashboard_type="invalid",
            report_name="x",
            file_name="x",
            dashboard_payload={},
        )

