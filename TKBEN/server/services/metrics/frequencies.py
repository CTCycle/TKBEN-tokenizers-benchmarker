from __future__ import annotations

import os
import tempfile
from collections import Counter
from collections.abc import Iterator

from sqlalchemy import Integer, String, create_engine, func, select
from sqlalchemy.orm import Session, declarative_base, mapped_column, sessionmaker

Base = declarative_base()


###############################################################################
class FrequencyEntry(Base):
    __tablename__ = "frequencies"
    token = mapped_column(String, primary_key=True, nullable=False)
    count = mapped_column(Integer, nullable=False, default=0, index=True)


###############################################################################
class DiskBackedFrequencyStore:
    def __init__(self, memory_limit: int = 250_000) -> None:
        self.memory_limit = max(10_000, int(memory_limit))
        self.memory: Counter[str] = Counter()
        fd, self.path = tempfile.mkstemp(prefix="tkben_freq_", suffix=".sqlite3")
        os.close(fd)
        self.engine = create_engine(f"sqlite:///{self.path}", future=True)
        self._session_factory = sessionmaker(bind=self.engine, future=True)
        self._initialize()

    # -------------------------------------------------------------------------
    def _initialize(self) -> None:
        Base.metadata.create_all(self.engine, checkfirst=True)

    # -------------------------------------------------------------------------
    def _session(self) -> Session:
        return self._session_factory()

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
        with self._session() as session:
            tokens = list(self.memory.keys())
            existing_rows = session.execute(
                select(FrequencyEntry).where(FrequencyEntry.token.in_(tokens))
            ).scalars().all()
            existing = {row.token: row for row in existing_rows}
            for token, count in self.memory.items():
                row = existing.get(token)
                if row is None:
                    session.add(FrequencyEntry(token=token, count=int(count)))
                else:
                    row.count = int(row.count) + int(count)
            session.commit()
        self.memory.clear()

    # -------------------------------------------------------------------------
    def total_count(self) -> int:
        self.flush()
        stmt = select(func.coalesce(func.sum(FrequencyEntry.count), 0))
        with self._session() as session:
            value = session.execute(stmt).scalar_one_or_none() or 0
        return int(value)

    # -------------------------------------------------------------------------
    def unique_count(self) -> int:
        self.flush()
        stmt = select(func.count(FrequencyEntry.token))
        with self._session() as session:
            value = session.execute(stmt).scalar_one_or_none() or 0
        return int(value)

    # -------------------------------------------------------------------------
    def token_counts(self) -> list[tuple[str, int]]:
        self.flush()
        stmt = select(FrequencyEntry.token, FrequencyEntry.count)
        with self._session() as session:
            rows = session.execute(stmt).all()
        return [(str(token), int(count)) for token, count in rows]

    # -------------------------------------------------------------------------
    def iter_counts(self, batch_size: int = 10_000):
        self.flush()
        stmt = select(FrequencyEntry.token, FrequencyEntry.count)
        with self._session() as session:
            result = session.execute(stmt)
            while True:
                rows = result.fetchmany(int(max(100, batch_size)))
                if not rows:
                    break
                for token, count in rows:
                    yield str(token), int(count)

    # -------------------------------------------------------------------------
    def iter_sorted_counts(
        self,
        descending: bool = True,
        batch_size: int = 10_000,
    ) -> Iterator[tuple[str, int]]:
        self.flush()
        order_count = FrequencyEntry.count.desc() if descending else FrequencyEntry.count.asc()
        stmt = select(FrequencyEntry.token, FrequencyEntry.count).order_by(
            order_count,
            FrequencyEntry.token.asc(),
        )
        with self._session() as session:
            result = session.execute(stmt)
            while True:
                rows = result.fetchmany(int(max(100, batch_size)))
                if not rows:
                    break
                for token, count in rows:
                    yield str(token), int(count)

    # -------------------------------------------------------------------------
    def top_k(self, k: int) -> list[tuple[str, int]]:
        self.flush()
        stmt = (
            select(FrequencyEntry.token, FrequencyEntry.count)
            .order_by(FrequencyEntry.count.desc(), FrequencyEntry.token.asc())
            .limit(int(max(1, k)))
        )
        with self._session() as session:
            rows = session.execute(stmt).all()
        return [(str(token), int(count)) for token, count in rows]

    # -------------------------------------------------------------------------
    def bottom_k(self, k: int) -> list[tuple[str, int]]:
        self.flush()
        stmt = (
            select(FrequencyEntry.token, FrequencyEntry.count)
            .order_by(FrequencyEntry.count.asc(), FrequencyEntry.token.asc())
            .limit(int(max(1, k)))
        )
        with self._session() as session:
            rows = session.execute(stmt).all()
        return [(str(token), int(count)) for token, count in rows]

    # -------------------------------------------------------------------------
    def sum_top_k(self, k: int) -> int:
        self.flush()
        limited = (
            select(FrequencyEntry.count)
            .order_by(FrequencyEntry.count.desc(), FrequencyEntry.token.asc())
            .limit(int(max(1, k)))
            .subquery()
        )
        stmt = select(func.coalesce(func.sum(limited.c.count), 0))
        with self._session() as session:
            value = session.execute(stmt).scalar_one_or_none() or 0
        return int(value)

    # -------------------------------------------------------------------------
    def sum_bottom_k(self, k: int) -> int:
        self.flush()
        limited = (
            select(FrequencyEntry.count)
            .order_by(FrequencyEntry.count.asc(), FrequencyEntry.token.asc())
            .limit(int(max(1, k)))
            .subquery()
        )
        stmt = select(func.coalesce(func.sum(limited.c.count), 0))
        with self._session() as session:
            value = session.execute(stmt).scalar_one_or_none() or 0
        return int(value)

    # -------------------------------------------------------------------------
    def longest_k(self, k: int) -> list[tuple[str, int]]:
        self.flush()
        stmt = (
            select(FrequencyEntry.token, FrequencyEntry.count)
            .order_by(func.length(FrequencyEntry.token).desc(), FrequencyEntry.token.asc())
            .limit(int(max(1, k)))
        )
        with self._session() as session:
            rows = session.execute(stmt).all()
        return [(str(token), int(count)) for token, count in rows]

    # -------------------------------------------------------------------------
    def shortest_k(self, k: int) -> list[tuple[str, int]]:
        self.flush()
        stmt = (
            select(FrequencyEntry.token, FrequencyEntry.count)
            .order_by(func.length(FrequencyEntry.token).asc(), FrequencyEntry.token.asc())
            .limit(int(max(1, k)))
        )
        with self._session() as session:
            rows = session.execute(stmt).all()
        return [(str(token), int(count)) for token, count in rows]

    # -------------------------------------------------------------------------
    def count_frequency_of_frequency(self, n: int) -> int:
        self.flush()
        stmt = select(func.count(FrequencyEntry.token)).where(
            FrequencyEntry.count == int(n)
        )
        with self._session() as session:
            value = session.execute(stmt).scalar_one_or_none() or 0
        return int(value)

    # -------------------------------------------------------------------------
    def close(self) -> None:
        try:
            self.engine.dispose()
        finally:
            try:
                os.remove(self.path)
            except OSError:
                pass
