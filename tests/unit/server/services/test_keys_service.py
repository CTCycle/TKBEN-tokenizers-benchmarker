from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine

from TKBEN.server.repositories.database.backend import database
from TKBEN.server.repositories.schemas.models import Base
from TKBEN.server.services.keys import HFAccessKeyService, HFAccessKeyValidationError


@pytest.fixture
def isolated_engine(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    monkeypatch.setattr(database.backend, "engine", engine)
    try:
        yield engine
    finally:
        engine.dispose()


def test_get_revealed_key_returns_decrypted_value(isolated_engine) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    inserted = service.repository.insert_key(
        encrypted_value="enc:secret-key",
        created_at=datetime.now(timezone.utc),
    )

    class FakeCipher:
        def decrypt(self, encrypted_value: str) -> str:
            if encrypted_value.startswith("enc:"):
                return encrypted_value[4:]
            raise ValueError("bad")

    service._cipher = FakeCipher()  # type: ignore[assignment]

    revealed = service.get_revealed_key(int(inserted.id))
    assert revealed == "secret-key"


def test_get_revealed_key_raises_validation_on_undecryptable(isolated_engine) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    inserted = service.repository.insert_key(
        encrypted_value="broken-value",
        created_at=datetime.now(timezone.utc),
    )

    class FailingCipher:
        def decrypt(self, encrypted_value: str) -> str:
            del encrypted_value
            raise ValueError("bad")

    service._cipher = FailingCipher()  # type: ignore[assignment]

    with pytest.raises(HFAccessKeyValidationError, match="cannot be decrypted"):
        service.get_revealed_key(int(inserted.id))
