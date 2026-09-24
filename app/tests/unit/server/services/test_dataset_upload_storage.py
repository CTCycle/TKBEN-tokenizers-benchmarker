from __future__ import annotations

import sqlite3
from typing import Any

from datasets import Dataset as HuggingFaceDataset
import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from server.repositories.database.backend import get_database
from server.repositories.database.sqlite import SQLiteRepository
from server.repositories.schemas.models import Base, Dataset, DatasetDocument
from server.services.datasets import DatasetService


def _build_service(monkeypatch: pytest.MonkeyPatch) -> tuple[DatasetService, Engine]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    event.listen(engine, "connect", SQLiteRepository.enable_foreign_keys)
    Base.metadata.create_all(engine, checkfirst=True)
    backend = get_database().backend
    monkeypatch.setattr(backend, "engine", engine)
    monkeypatch.setattr(
        backend,
        "session_factory",
        sessionmaker(bind=engine, future=True),
    )
    service = DatasetService()
    service.streaming_batch_size = 1
    service.log_interval = 1
    return service, engine


def test_near_limit_csv_import_persists_large_text(monkeypatch) -> None:
    service, engine = _build_service(monkeypatch)
    try:
        limit = service.settings.max_upload_bytes
        near_limit_size = limit - 1024 * 1024
        header = b"text\n"
        file_content = header + b"x" * (near_limit_size - len(header))

        result = service.upload_and_persist(
            file_content=file_content,
            filename="near-limit.csv",
        )

        assert result["dataset_name"] == "custom/near-limit"
        assert result["document_count"] == 1
        assert result["saved_count"] == 1
        with Session(bind=engine) as session:
            dataset = session.scalar(
                select(Dataset).where(Dataset.name == "custom/near-limit")
            )
            assert dataset is not None
            text_length = session.scalar(
                select(func.length(DatasetDocument.text)).where(
                    DatasetDocument.dataset_id == dataset.id
                )
            )

        assert dataset.status == "ready"
        assert text_length == near_limit_size - len(header)
    finally:
        engine.dispose()


@pytest.mark.parametrize("import_kind", ["upload", "download"])
def test_sqlite_full_cleans_partial_import_and_retry_succeeds(
    monkeypatch,
    import_kind: str,
) -> None:
    service, engine = _build_service(monkeypatch)
    original_save_batch = service.dataset_repository.save_document_batch
    saved_batches = 0
    dataset_name = (
        "custom/recoverable" if import_kind == "upload" else "public/recoverable"
    )

    def save_batch_and_exhaust_database(batch: list[dict[str, Any]]) -> None:
        nonlocal saved_batches
        original_save_batch(batch)
        saved_batches += 1
        if saved_batches == 1:
            with engine.connect() as connection:
                page_count = connection.exec_driver_sql("PRAGMA page_count").scalar_one()
                connection.exec_driver_sql(f"PRAGMA max_page_count={page_count}")
                connection.commit()

    monkeypatch.setattr(
        service.dataset_repository,
        "save_document_batch",
        save_batch_and_exhaust_database,
    )

    def persist_import() -> int:
        if import_kind == "upload":
            result = service.upload_and_persist(
                file_content=b"text\nfirst\n" + b"x" * 100_000,
                filename="recoverable.csv",
            )
            return result["saved_count"]

        dataset = HuggingFaceDataset.from_dict(
            {"text": ["first", "x" * 100_000]}
        )
        stats = service.collect_length_statistics(
            service.dataset_length_stream(dataset, "text", remove_invalid=True)
        )
        _, saved_count = service.persist_dataset(
            dataset=dataset,
            dataset_name=dataset_name,
            text_column="text",
            stats=stats,
            remove_invalid=True,
        )
        return saved_count

    try:
        with pytest.raises(OperationalError) as error:
            persist_import()

        assert error.value.orig.sqlite_errorcode == sqlite3.SQLITE_FULL
        with Session(bind=engine) as session:
            leftover = session.scalar(
                select(Dataset).where(Dataset.name == dataset_name)
            )
            leftover_documents = session.scalar(
                select(func.count(DatasetDocument.id))
            )
        assert leftover is None
        assert leftover_documents == 0

        with engine.connect() as connection:
            connection.exec_driver_sql("PRAGMA max_page_count=2147483646")
            connection.commit()
        monkeypatch.setattr(
            service.dataset_repository,
            "save_document_batch",
            original_save_batch,
        )

        assert persist_import() == 2
        with Session(bind=engine) as session:
            dataset = session.scalar(
                select(Dataset).where(Dataset.name == dataset_name)
            )
            assert dataset is not None
            document_count = session.scalar(
                select(func.count(DatasetDocument.id)).where(
                    DatasetDocument.dataset_id == dataset.id
                )
            )

        assert dataset.status == "ready"
        assert document_count == 2
    finally:
        engine.dispose()
