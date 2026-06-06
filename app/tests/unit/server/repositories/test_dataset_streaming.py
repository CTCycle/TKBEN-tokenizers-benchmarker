from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from server.repositories.database.backend import get_database
from server.repositories.schemas.models import Base, Dataset, DatasetDocument
from server.repositories.serialization.data import DatasetSerializer


def test_streaming_preserves_empty_and_unicode_rows(monkeypatch) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)

    with Session(bind=engine) as session:
        dataset = Dataset(name="custom/stream")
        session.add(dataset)
        session.flush()
        session.add_all(
            [
                DatasetDocument(dataset_id=dataset.id, text=""),
                DatasetDocument(dataset_id=dataset.id, text=" "),
                DatasetDocument(dataset_id=dataset.id, text="emoji 😀"),
                DatasetDocument(dataset_id=dataset.id, text="CJK 漢字"),
            ]
        )
        session.commit()

    serializer = DatasetSerializer()
    rows = list(
        serializer.iterate_dataset_rows_for_benchmarks("custom/stream", batch_size=2)
    )
    assert [text for _, text in rows] == ["", " ", "emoji 😀", "CJK 漢字"]
