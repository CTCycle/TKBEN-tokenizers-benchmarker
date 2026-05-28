from __future__ import annotations

from server.domain.benchmark_observations import TokenizerRunConfig
from server.services.benchmark_engine import run_tokenizer_trials
from server.services.tokenizer_adapters import EncodedBatch


class MemoryAdapter:
    tokenizer_id = "dummy"

    def encode_batch(self, texts, **kwargs) -> EncodedBatch:  # type: ignore[no-untyped-def]
        del kwargs
        return EncodedBatch(
            token_counts=[len(text) for text in texts],
            unknown_counts=[0 for _ in texts],
            input_ids_by_doc=[[ord(character) for character in text] for text in texts],
        )


def test_memory_peak_is_measured_not_placeholder() -> None:
    observations = run_tokenizer_trials(
        tokenizer=MemoryAdapter(),
        text_batches_factory=lambda: [["a", "b"]],
        config=TokenizerRunConfig(batch_size=2),
        warmup_trials=0,
        timed_trials=1,
    )
    assert observations
    assert observations[0].peak_rss_mb is not None
    assert float(observations[0].peak_rss_mb or 0.0) > 0.0
