from __future__ import annotations

from server.services.benchmark_streams import (
    iter_limited_rows,
    iter_text_batches,
    normalize_benchmark_text,
)

###############################################################################
def test_normalize_benchmark_text_handles_none_and_coercion() -> None:
    assert normalize_benchmark_text(None) is None
    assert normalize_benchmark_text("abc") == "abc"
    assert normalize_benchmark_text(123) == "123"

###############################################################################
def test_iter_limited_rows_respects_limit_and_skips_none() -> None:
    rows = [(1, "a"), (2, None), (3, "b"), (4, "c")]
    limited = list(iter_limited_rows(rows, 3))
    assert limited == [(1, "a"), (3, "b")]

###############################################################################
def test_iter_text_batches_groups_rows_by_batch_size() -> None:
    rows = [(1, "a"), (2, "b"), (3, "c")]
    batches = list(iter_text_batches(rows, 2))
    assert batches == [[(1, "a"), (2, "b")], [(3, "c")]]
