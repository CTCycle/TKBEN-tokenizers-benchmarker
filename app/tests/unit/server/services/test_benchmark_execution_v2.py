from __future__ import annotations

from typing import Any

import TKBEN.server.services.benchmarks as benchmarks_module
from TKBEN.server.domain.benchmarks import BenchmarkRunResponse
from TKBEN.server.services.benchmarks import BenchmarkService


class DummyTokenizer:
    name_or_path = "dummy/tokenizer"

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {
            "alpha": 1,
            "beta": 2,
            "gamma": 3,
            "delta": 4,
            "##tail": 5,
        }
        self._id_to_token = {value: key for key, value in self._vocab.items()}

    def tokenize(self, text: str) -> list[str]:
        return str(text).split()

    def encode(self, text: str) -> list[int]:
        return [self._vocab.get(token, 0) for token in str(text).split()]

    def decode(self, token_ids: Any) -> str:
        ids = token_ids.ids if hasattr(token_ids, "ids") else token_ids
        return " ".join(self._id_to_token.get(int(token_id), "[UNK]") for token_id in ids)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self._id_to_token.get(int(token_id), "[UNK]") for token_id in token_ids]

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def test_run_benchmarks_returns_v2_contract() -> None:
    service = BenchmarkService()
    rows = [
        (10, "alpha beta beta"),
        (11, "alpha gamma"),
        (12, "delta"),
    ]

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {"dummy/tokenizer": DummyTokenizer()}  # type: ignore[method-assign]
    service.calculate_morphological_consistency = (  # type: ignore[method-assign]
        lambda tokenizer, base_words: 0.5
    )

    original_perf_counter = benchmarks_module.time.perf_counter
    counter = {"calls": 0}

    def fake_perf_counter() -> float:
        counter["calls"] += 1
        return 100.0 if counter["calls"] == 1 else 104.0

    benchmarks_module.time.perf_counter = fake_perf_counter
    try:
        result = service.run_benchmarks(
            dataset_name="custom/ds",
            tokenizer_ids=["dummy/tokenizer"],
            selected_metric_keys=None,
        )
    finally:
        benchmarks_module.time.perf_counter = original_perf_counter

    assert isinstance(result, BenchmarkRunResponse)
    assert result.status == "success"
    assert result.dataset_name == "custom/ds"
    assert result.documents_processed == 3
    assert result.tokenizers_count == 1
    assert len(result.tokenizer_results) == 1
    assert len(result.chart_data.efficiency) == 1
    assert len(result.per_document_stats) == 1
    assert result.tokenizer_results[0].tokenizer == "dummy/tokenizer"
    assert result.per_document_stats[0].tokenizer == "dummy/tokenizer"
