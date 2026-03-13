from __future__ import annotations

import os
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator

from TKBEN.server.repositories.queries import frequencies as frequency_queries


###############################################################################
class DiskBackedFrequencyStore:
    def __init__(self, memory_limit: int = 250_000) -> None:
        self.memory_limit = max(10_000, int(memory_limit))
        self.memory: Counter[str] = Counter()
        fd, self.path = tempfile.mkstemp(prefix="tkben_freq_", suffix=".sqlite3")
        os.close(fd)
        self.conn = sqlite3.connect(self.path)
        self._initialize()

    # -------------------------------------------------------------------------
    def _initialize(self) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute(frequency_queries.CREATE_FREQUENCIES_TABLE)
            cursor.execute(frequency_queries.CREATE_FREQUENCIES_COUNT_INDEX)
            self.conn.commit()
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def add(self, token: str, count: int = 1) -> None:
        if not token:
            return
        self.memory[str(token)] += int(count)
        if len(self.memory) >= self.memory_limit:
            self.flush()

    # -------------------------------------------------------------------------
    def add_many(self, tokens: list[str]) -> None:
        for token in tokens:
            self.add(token, 1)

    # -------------------------------------------------------------------------
    def flush(self) -> None:
        if not self.memory:
            return
        cursor = self.conn.cursor()
        try:
            for token, count in self.memory.items():
                cursor.execute(
                    frequency_queries.UPSERT_FREQUENCY_COUNT,
                    (token, int(count)),
                )
            self.conn.commit()
            self.memory.clear()
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def total_count(self) -> int:
        self.flush()
        cursor = self.conn.cursor()
        try:
            row = cursor.execute(frequency_queries.SELECT_TOTAL_FREQUENCY_COUNT).fetchone()
            return int(row[0] if row else 0)
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def unique_count(self) -> int:
        self.flush()
        cursor = self.conn.cursor()
        try:
            row = cursor.execute(frequency_queries.SELECT_UNIQUE_FREQUENCY_COUNT).fetchone()
            return int(row[0] if row else 0)
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def token_counts(self) -> list[tuple[str, int]]:
        self.flush()
        cursor = self.conn.cursor()
        try:
            rows = cursor.execute(frequency_queries.SELECT_ALL_TOKEN_COUNTS).fetchall()
            return [(str(token), int(count)) for token, count in rows]
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def iter_counts(self, batch_size: int = 10_000):
        self.flush()
        cursor = self.conn.cursor()
        try:
            cursor.execute(frequency_queries.SELECT_ALL_TOKEN_COUNTS)
            while True:
                rows = cursor.fetchmany(int(max(100, batch_size)))
                if not rows:
                    break
                for token, count in rows:
                    yield str(token), int(count)
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def iter_sorted_counts(
        self,
        descending: bool = True,
        batch_size: int = 10_000,
    ) -> Iterator[tuple[str, int]]:
        self.flush()
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                frequency_queries.select_sorted_token_counts_query(
                    descending=descending
                )
            )
            while True:
                rows = cursor.fetchmany(int(max(100, batch_size)))
                if not rows:
                    break
                for token, count in rows:
                    yield str(token), int(count)
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def top_k(self, k: int) -> list[tuple[str, int]]:
        self.flush()
        cursor = self.conn.cursor()
        try:
            rows = cursor.execute(
                frequency_queries.SELECT_TOP_K_TOKEN_COUNTS,
                (int(max(1, k)),),
            ).fetchall()
            return [(str(token), int(count)) for token, count in rows]
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def bottom_k(self, k: int) -> list[tuple[str, int]]:
        self.flush()
        cursor = self.conn.cursor()
        try:
            rows = cursor.execute(
                frequency_queries.SELECT_BOTTOM_K_TOKEN_COUNTS,
                (int(max(1, k)),),
            ).fetchall()
            return [(str(token), int(count)) for token, count in rows]
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def sum_top_k(self, k: int) -> int:
        self.flush()
        cursor = self.conn.cursor()
        try:
            row = cursor.execute(
                frequency_queries.SELECT_SUM_TOP_K_COUNTS,
                (int(max(1, k)),),
            ).fetchone()
            return int(row[0] if row else 0)
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def sum_bottom_k(self, k: int) -> int:
        self.flush()
        cursor = self.conn.cursor()
        try:
            row = cursor.execute(
                frequency_queries.SELECT_SUM_BOTTOM_K_COUNTS,
                (int(max(1, k)),),
            ).fetchone()
            return int(row[0] if row else 0)
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def longest_k(self, k: int) -> list[tuple[str, int]]:
        self.flush()
        cursor = self.conn.cursor()
        try:
            rows = cursor.execute(
                frequency_queries.SELECT_LONGEST_K_TOKEN_COUNTS,
                (int(max(1, k)),),
            ).fetchall()
            return [(str(token), int(count)) for token, count in rows]
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def shortest_k(self, k: int) -> list[tuple[str, int]]:
        self.flush()
        cursor = self.conn.cursor()
        try:
            rows = cursor.execute(
                frequency_queries.SELECT_SHORTEST_K_TOKEN_COUNTS,
                (int(max(1, k)),),
            ).fetchall()
            return [(str(token), int(count)) for token, count in rows]
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def count_frequency_of_frequency(self, n: int) -> int:
        self.flush()
        cursor = self.conn.cursor()
        try:
            row = cursor.execute(
                frequency_queries.SELECT_FREQUENCY_OF_FREQUENCY_COUNT,
                (int(n),),
            ).fetchone()
            return int(row[0] if row else 0)
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def close(self) -> None:
        try:
            self.conn.close()
        finally:
            try:
                os.remove(self.path)
            except OSError:
                pass
