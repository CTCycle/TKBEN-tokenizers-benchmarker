from __future__ import annotations

from server.domain.benchmark_observations import TokenizerRunConfig
from server.services.benchmark_engine import run_tokenizer_trials
from server.services.tokenizer_adapters import EncodedBatch


class CountingAdapter:
    tokenizer_id = "dummy"

    def __init__(self) -> None:
        self.calls = 0

    def encode_batch(self, texts, **kwargs) -> EncodedBatch:  # type: ignore[no-untyped-def]
        del kwargs
        self.calls += 1
        return EncodedBatch(token_counts=[len(text) for text in texts], unknown_counts=[0 for _ in texts])


def test_warmup_excluded_and_timed_trials_control_observations() -> None:
    adapter = CountingAdapter()
    batches = [["a", "bb"], ["ccc"]]
    observations = run_tokenizer_trials(
        tokenizer=adapter,
        text_batches_factory=lambda: batches,
        config=TokenizerRunConfig(batch_size=2),
        warmup_trials=2,
        timed_trials=3,
    )
    assert len(observations) == 6
    assert adapter.calls == (2 * 2) + (3 * 2)
