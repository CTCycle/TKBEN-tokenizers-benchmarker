from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice


###############################################################################
def normalize_benchmark_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


###############################################################################
def iter_limited_rows(
    rows: Iterable[tuple[int, object]],
    max_documents: int | None,
) -> Iterator[tuple[int, str]]:
    limited_rows: Iterable[tuple[int, object]] = rows
    if isinstance(max_documents, int) and max_documents > 0:
        limited_rows = islice(rows, max_documents)

    for row_id, value in limited_rows:
        text = normalize_benchmark_text(value)
        if text is None:
            continue
        yield int(row_id), text


###############################################################################
def iter_text_batches(
    rows: Iterable[tuple[int, str]],
    batch_size: int,
) -> Iterator[list[tuple[int, str]]]:
    size = max(1, int(batch_size))
    batch: list[tuple[int, str]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
