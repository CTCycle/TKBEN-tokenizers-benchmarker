from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from TKBEN.server.repositories.database.backend import database
from TKBEN.server.repositories.schemas.models import Base, HFAccessKey
from TKBEN.server.services.keys import (
    HFAccessKeyConflictError,
    HFAccessKeyNotFoundError,
    HFAccessKeyService,
    HFAccessKeyValidationError,
)


###############################################################################
@pytest.fixture
def isolated_engine(monkeypatch: pytest.MonkeyPatch):
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    monkeypatch.setattr(database.backend, "engine", engine)
    try:
        yield engine
    finally:
        engine.dispose()


###############################################################################
def test_get_active_key_raises_validation_error_on_invalid_decryption(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    class FailingCipher:
        def decrypt(self, encrypted_value: str) -> str:
            del encrypted_value
            raise ValueError("invalid token")

    with Session(bind=database.backend.engine) as session:
        session.add(
            HFAccessKey(
                key_value="encrypted-value",
                created_at=datetime.now(timezone.utc),
                is_active=True,
            )
        )
        session.commit()

    service._cipher = FailingCipher()  # type: ignore[assignment]
    with pytest.raises(HFAccessKeyValidationError, match="cannot be decrypted"):
        service.get_active_key()


###############################################################################
def test_add_key_skips_undecryptable_rows_during_duplicate_check(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    class RecoveringCipher:
        def encrypt(self, plaintext: str) -> str:
            return f"enc:{plaintext}"

        def decrypt(self, encrypted_value: str) -> str:
            if encrypted_value == "stale-encrypted":
                raise ValueError("invalid token")
            if encrypted_value.startswith("enc:"):
                return encrypted_value[4:]
            return encrypted_value

    with Session(bind=database.backend.engine) as session:
        session.add(
            HFAccessKey(
                key_value="stale-encrypted",
                created_at=datetime.now(timezone.utc),
                is_active=False,
            )
        )
        session.commit()

    service._cipher = RecoveringCipher()  # type: ignore[assignment]
    result = service.add_key("hf_test_key")
    assert result["is_active"] is False

    with Session(bind=database.backend.engine) as session:
        rows = session.execute(
            select(HFAccessKey).order_by(HFAccessKey.id.asc())
        ).scalars().all()
    assert len(rows) == 2
    assert rows[1].key_value == "enc:hf_test_key"


###############################################################################
def test_get_active_key_migrates_legacy_plaintext_key(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    class LegacyAwareCipher:
        def encrypt(self, plaintext: str) -> str:
            return f"enc:{plaintext}"

        def decrypt(self, encrypted_value: str) -> str:
            if encrypted_value.startswith("enc:"):
                return encrypted_value[4:]
            raise ValueError("invalid token")

    with Session(bind=database.backend.engine) as session:
        session.add(
            HFAccessKey(
                key_value="hf_legacy_token_123",
                created_at=datetime.now(timezone.utc),
                is_active=True,
            )
        )
        session.commit()

    service._cipher = LegacyAwareCipher()  # type: ignore[assignment]
    key = service.get_active_key()
    assert key == "hf_legacy_token_123"

    with Session(bind=database.backend.engine) as session:
        stored = session.execute(
            select(HFAccessKey).where(HFAccessKey.is_active.is_(True)).limit(1)
        ).scalar_one()
    assert stored.key_value == "enc:hf_legacy_token_123"


###############################################################################
def test_add_key_detects_duplicate_for_legacy_plaintext_row(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    class LegacyAwareCipher:
        def encrypt(self, plaintext: str) -> str:
            return f"enc:{plaintext}"

        def decrypt(self, encrypted_value: str) -> str:
            if encrypted_value.startswith("enc:"):
                return encrypted_value[4:]
            raise ValueError("invalid token")

    with Session(bind=database.backend.engine) as session:
        session.add(
            HFAccessKey(
                key_value="hf_legacy_token_123",
                created_at=datetime.now(timezone.utc),
                is_active=False,
            )
        )
        session.commit()

    service._cipher = LegacyAwareCipher()  # type: ignore[assignment]
    with pytest.raises(HFAccessKeyConflictError, match="already stored"):
        service.add_key("hf_legacy_token_123")


###############################################################################
def test_set_active_key_is_idempotent_for_already_active_key(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    with Session(bind=database.backend.engine) as session:
        session.add_all(
            [
                HFAccessKey(
                    key_value="enc:key-1",
                    created_at=datetime.now(timezone.utc),
                    is_active=True,
                ),
                HFAccessKey(
                    key_value="enc:key-2",
                    created_at=datetime.now(timezone.utc),
                    is_active=False,
                ),
            ]
        )
        session.commit()
        key_id = int(
            session.execute(
                select(HFAccessKey.id).where(HFAccessKey.key_value == "enc:key-1")
            ).scalar_one()
        )

    service.set_active_key(key_id)

    with Session(bind=database.backend.engine) as session:
        rows = session.execute(select(HFAccessKey).order_by(HFAccessKey.id.asc())).scalars().all()
    assert any(row.id == key_id and row.is_active for row in rows)
    assert all(row.is_active is (row.id == key_id) for row in rows)


###############################################################################
def test_set_active_key_raises_not_found_for_unknown_key(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()
    with pytest.raises(HFAccessKeyNotFoundError, match="not found"):
        service.set_active_key(404)
