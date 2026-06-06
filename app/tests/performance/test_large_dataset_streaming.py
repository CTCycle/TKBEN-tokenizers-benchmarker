from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from server.repositories.database.backend import get_database
from server.repositories.schemas.models import Base, Dataset, DatasetDocument
from server.repositories.serialization.data import DatasetSerializer


def test_large_dataset_streaming_batches_do_not_materialize_all_rows(
    monkeypatch,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)
    with Session(bind=engine) as session:
        dataset = Dataset(name="custom/large_stream")
        session.add(dataset)
        session.flush()
        session.add_all(
            [
                DatasetDocument(dataset_id=dataset.id, text=f"row-{index}")
                for index in range(2000)
            ]
        )
        session.commit()

    serializer = DatasetSerializer()
    batches = list(
        serializer.iterate_dataset_batches("custom/large_stream", batch_size=128)
    )
    assert len(batches) >= 10
    assert sum(len(batch) for batch in batches) == 2000
