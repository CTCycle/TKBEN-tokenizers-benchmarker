from __future__ import annotations

from typing import Any

import pytest

from TKBEN.server.repositories.serialization.data import TokenizerReportSerializer
from TKBEN.server.services.tokenizers import TokenizersService


def test_compute_subword_word_stats_excludes_special_tokens_and_classifies_markers() -> None:
    service = TokenizersService()
    stats = service.compute_subword_word_stats(
        vocab_tokens=[
            "##ing",
            "token",
            "▁hello",
            "he▁llo",
            "Ġword",
            "wordĠpiece",
            "sub@@",
            "Ċnewline",
            "lineĊbreak",
            "[CLS]",
            "<pad>",
        ],
        special_tokens={"[CLS]", "<pad>"},
    )

    assert stats["subword_count"] == 5
    assert stats["word_count"] == 4
    assert stats["considered_count"] == 9
    assert stats["subword_to_word_ratio"] == pytest.approx(1.25)
    assert stats["subword_percentage"] == pytest.approx(55.5555, rel=1e-3)
    assert stats["word_percentage"] == pytest.approx(44.4444, rel=1e-3)


def test_resolve_hf_repo_metadata_returns_link_when_description_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TokenizersService()

    def raise_model_info(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("model info unavailable")

    def raise_model_card(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("model card unavailable")

    monkeypatch.setattr(
        "TKBEN.server.services.tokenizers.HfApi.model_info",
        raise_model_info,
    )
    monkeypatch.setattr(
        "TKBEN.server.services.tokenizers.ModelCard.load",
        raise_model_card,
    )

    description, huggingface_url = service.resolve_hf_repo_metadata("bert-base-uncased")

    assert description is None
    assert huggingface_url == "https://huggingface.co/bert-base-uncased"


class DummyBackendTokenizerModel:
    pass


class DummyBackendTokenizer:
    def __init__(self) -> None:
        self.model = DummyBackendTokenizerModel()


class DummyTokenizer:
    special_tokens_map = {"cls_token": "[CLS]"}
    all_special_tokens = ["[CLS]", "<pad>"]
    all_special_ids = [2, 99]
    added_tokens_encoder = {"<pad>": 99}
    vocab_size = 30_522
    model_max_length = 512
    padding_side = "right"
    init_kwargs = {
        "model_max_length": 512,
        "padding_side": "right",
    }

    backend_tokenizer = DummyBackendTokenizer()

    def get_vocab(self) -> dict[str, int]:
        return {
            "hello": 0,
            "##ing": 1,
            "[CLS]": 2,
            "▁world": 3,
            "he▁llo": 4,
            "Ġtoken": 5,
            "wordĠpiece": 6,
        }


def test_generate_report_payload_includes_hf_url_and_subword_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TokenizersService()
    captured_report: dict[str, Any] = {}

    monkeypatch.setattr(service, "has_cached_tokenizer", lambda tokenizer_name: True)
    monkeypatch.setattr(
        "TKBEN.server.services.tokenizers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    monkeypatch.setattr(service, "find_cached_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "load_json_if_present", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        service.report_serializer,
        "replace_tokenizer_vocabulary",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        service,
        "resolve_hf_repo_metadata",
        lambda tokenizer_name: (None, f"https://huggingface.co/{tokenizer_name}"),
    )

    def capture_report(report: dict[str, Any]) -> int:
        captured_report.update(report)
        return 123

    monkeypatch.setattr(service.report_serializer, "save_tokenizer_report", capture_report)

    report = service.generate_and_store_report("bert-base-uncased")

    assert report["report_id"] == 123
    assert report["huggingface_url"] == "https://huggingface.co/bert-base-uncased"
    assert report["global_stats"]["persistence_mode"] == "filesystem_required"
    assert "subword_word_stats" in report["global_stats"]
    assert report["global_stats"]["base_vocabulary_size"] == 30522
    assert report["global_stats"]["model_max_length"] == 512
    assert report["global_stats"]["padding_side"] == "right"
    assert report["global_stats"]["added_tokens_count"] == 1
    assert report["global_stats"]["special_tokens_ids_count"] == 2
    assert report["global_stats"]["special_tokens_count"] == 2
    assert report["global_stats"]["vocabulary_stats"]["subword_like_count"] == 3
    assert report["global_stats"]["vocabulary_stats"]["special_tokens_in_vocab_count"] == 1
    assert captured_report["huggingface_url"] == "https://huggingface.co/bert-base-uncased"
    assert captured_report["global_stats"]["vocabulary_stats"]["unique_token_lengths"] == 3


def test_tokenizer_report_serializer_roundtrip_preserves_huggingface_url() -> None:
    serializer = TokenizerReportSerializer()
    report_id = serializer.save_tokenizer_report(
        {
            "report_version": 1,
            "created_at": "2026-02-17T00:00:00Z",
            "tokenizer_name": "test/tokenizer-report-roundtrip",
            "description": None,
            "huggingface_url": "https://huggingface.co/test/tokenizer-report-roundtrip",
            "global_stats": {
                "vocabulary_size": 2,
                "base_vocabulary_size": 2,
                "model_max_length": 512,
                "padding_side": "right",
                "added_tokens_count": 0,
                "special_tokens_ids_count": 1,
                "subword_word_stats": {
                    "heuristic": "hash_prefix_atat_suffix_sentencepiece_markers",
                    "subword_count": 1,
                    "word_count": 1,
                    "considered_count": 2,
                    "subword_percentage": 50.0,
                    "word_percentage": 50.0,
                    "subword_to_word_ratio": 1.0,
                },
                "vocabulary_stats": {
                    "heuristic": "hash_prefix_atat_suffix_sentencepiece_markers",
                    "min_token_length": 1,
                    "mean_token_length": 1.0,
                    "median_token_length": 1.0,
                    "max_token_length": 1,
                    "subword_like_count": 1,
                    "subword_like_percentage": 50.0,
                    "special_tokens_in_vocab_count": 0,
                    "special_tokens_in_vocab_percentage": 0.0,
                    "unique_token_lengths": 1,
                    "empty_token_count": 0,
                    "considered_non_special_count": 2,
                },
                "persistence_mode": "filesystem_required",
                "persistence_reason": "test",
            },
            "token_length_histogram": {
                "bins": ["1-1"],
                "counts": [2],
                "bin_edges": [1.0, 2.0],
                "min_length": 1,
                "max_length": 1,
                "mean_length": 1.0,
                "median_length": 1.0,
            },
            "vocabulary_size": 2,
        }
    )

    loaded = serializer.load_tokenizer_report_by_id(report_id)

    assert loaded is not None
    assert loaded["huggingface_url"] == "https://huggingface.co/test/tokenizer-report-roundtrip"
    assert loaded["global_stats"]["subword_word_stats"]["subword_count"] == 1
    assert loaded["global_stats"]["base_vocabulary_size"] == 2
    assert loaded["global_stats"]["model_max_length"] == 512
    assert loaded["global_stats"]["vocabulary_stats"]["subword_like_count"] == 1
