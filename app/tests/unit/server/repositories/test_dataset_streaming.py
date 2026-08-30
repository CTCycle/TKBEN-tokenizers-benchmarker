from __future__ import annotations

from sqlalchemy import create_engine
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from server.repositories.database.backend import get_database
from server.repositories.schemas.models import Base, Dataset, DatasetDocument
from server.repositories.datasets import DatasetRepository


###############################################################################
def test_streaming_preserves_empty_and_unicode_rows(monkeypatch) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)

    with Session(bind=engine) as session:
        now = datetime.now(timezone.utc)
        dataset = Dataset(
            name="custom/stream",
            status="ready",
            created_at=now,
            updated_at=now,
            ready_at=now,
        )
        session.add(dataset)
        session.flush()
        session.add_all(
            [
                DatasetDocument(dataset_id=dataset.id, ordinal=0, text=""),
                DatasetDocument(dataset_id=dataset.id, ordinal=1, text=" "),
                DatasetDocument(dataset_id=dataset.id, ordinal=2, text="emoji 😀"),
                DatasetDocument(dataset_id=dataset.id, ordinal=3, text="CJK 漢字"),
            ]
        )
        session.commit()

    repository = DatasetRepository()
    rows = list(
        repository.iterate_dataset_rows_for_benchmarks("custom/stream", batch_size=2)
    )
    assert [text for _, text in rows] == ["", " ", "emoji 😀", "CJK 漢字"]
