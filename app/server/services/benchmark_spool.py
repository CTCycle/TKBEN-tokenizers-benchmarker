from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path

###############################################################################
class BenchmarkTextSpool:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        temp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".ndjson",
            delete=False,
        )
        self._path = Path(temp.name)
        self._handle = temp

    # -------------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._path

    # -------------------------------------------------------------------------
    def append(self, row_id: int, text: str) -> None:
        self._handle.write(
            json.dumps({"row_id": int(row_id), "text": str(text)}, ensure_ascii=False)
        )
        self._handle.write("\n")

    # -------------------------------------------------------------------------
    def finalize(self) -> None:
        self._handle.flush()
        self._handle.close()

    # -------------------------------------------------------------------------
    def iter_rows(self) -> Iterator[tuple[int, str]]:
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                yield int(payload["row_id"]), str(payload["text"])

    # -------------------------------------------------------------------------
    def iter_text_batches(self, batch_size: int) -> Iterator[list[str]]:
        size = max(1, int(batch_size))
        batch: list[str] = []
        for _, text in self.iter_rows():
            batch.append(text)
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch

    # -------------------------------------------------------------------------
    def cleanup(self) -> None:
        try:
            if self._path.exists():
                self._path.unlink()
        except OSError:
            pass
