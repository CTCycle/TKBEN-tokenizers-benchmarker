from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from server.repositories.database.backend import get_database
from server.repositories.schemas.models import Base, HFAccessKey
from server.services.keys import (
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
    database = get_database()
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

    ###############################################################################
    class FailingCipher:

        # -------------------------------------------------------------------------
        def decrypt(self, encrypted_value: str) -> str:
            del encrypted_value
            raise ValueError("invalid token")

    with Session(bind=get_database().backend.engine) as session:
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

    ###############################################################################
    class RecoveringCipher:

        # -------------------------------------------------------------------------
        def encrypt(self, plaintext: str) -> str:
            return f"enc:{plaintext}"

        # -------------------------------------------------------------------------
        def decrypt(self, encrypted_value: str) -> str:
            if encrypted_value == "stale-encrypted":
                raise ValueError("invalid token")
            if encrypted_value.startswith("enc:"):
                return encrypted_value[4:]
            return encrypted_value

    with Session(bind=get_database().backend.engine) as session:
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

    with Session(bind=get_database().backend.engine) as session:
        rows = (
            session.execute(select(HFAccessKey).order_by(HFAccessKey.id.asc()))
            .scalars()
            .all()
        )
    assert len(rows) == 2
    assert rows[1].key_value == "enc:hf_test_key"

###############################################################################
def test_add_key_detects_duplicate_raw_key_across_encryption(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    ###############################################################################
    class NonDeterministicCipher:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.encryptions = 0

        # -------------------------------------------------------------------------
        def encrypt(self, plaintext: str) -> str:
            self.encryptions += 1
            return f"enc:{self.encryptions}:{plaintext}"

        # -------------------------------------------------------------------------
        def decrypt(self, encrypted_value: str) -> str:
            return encrypted_value.rsplit(":", 1)[-1]

    service._cipher = NonDeterministicCipher()  # type: ignore[assignment]

    service.add_key("hf_duplicate_key")
    with pytest.raises(HFAccessKeyConflictError, match="already stored"):
        service.add_key("hf_duplicate_key")

    with Session(bind=get_database().backend.engine) as session:
        rows = session.execute(select(HFAccessKey)).scalars().all()
    assert len(rows) == 1

###############################################################################
def test_get_active_key_rejects_plaintext_legacy_value(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    ###############################################################################
    class StrictCipher:

        # -------------------------------------------------------------------------
        def decrypt(self, encrypted_value: str) -> str:
            if encrypted_value.startswith("enc:"):
                return encrypted_value[4:]
            raise ValueError("invalid token")

    with Session(bind=get_database().backend.engine) as session:
        session.add(
            HFAccessKey(
                key_value="hf_legacy_token_123",
                created_at=datetime.now(timezone.utc),
                is_active=True,
            )
        )
        session.commit()

    service._cipher = StrictCipher()  # type: ignore[assignment]
    with pytest.raises(HFAccessKeyValidationError, match="cannot be decrypted"):
        service.get_active_key()

###############################################################################
def test_set_active_key_is_idempotent_for_already_active_key(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    with Session(bind=get_database().backend.engine) as session:
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

    with Session(bind=get_database().backend.engine) as session:
        rows = (
            session.execute(select(HFAccessKey).order_by(HFAccessKey.id.asc()))
            .scalars()
            .all()
        )
    assert any(row.id == key_id and row.is_active for row in rows)
    assert all(row.is_active is (row.id == key_id) for row in rows)

###############################################################################
def test_activating_second_key_deactivates_first(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    with Session(bind=get_database().backend.engine) as session:
        session.add_all(
            [
                HFAccessKey(
                    key_value="enc:key-1",
                    created_at=datetime.now(timezone.utc),
                    is_active=False,
                ),
                HFAccessKey(
                    key_value="enc:key-2",
                    created_at=datetime.now(timezone.utc),
                    is_active=False,
                ),
            ]
        )
        session.commit()
        key_ids = list(
            session.execute(select(HFAccessKey.id).order_by(HFAccessKey.id.asc()))
            .scalars()
            .all()
        )

    service.set_active_key(int(key_ids[0]))
    service.set_active_key(int(key_ids[1]))

    with Session(bind=get_database().backend.engine) as session:
        rows = (
            session.execute(select(HFAccessKey).order_by(HFAccessKey.id.asc()))
            .scalars()
            .all()
        )
    assert rows[0].is_active is False
    assert rows[1].is_active is True

###############################################################################
def test_unknown_activation_does_not_clear_existing_active_key(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    with Session(bind=get_database().backend.engine) as session:
        session.add(
            HFAccessKey(
                key_value="enc:key-1",
                created_at=datetime.now(timezone.utc),
                is_active=True,
            )
        )
        session.commit()
        key_id = int(session.execute(select(HFAccessKey.id)).scalar_one())

    with pytest.raises(HFAccessKeyNotFoundError, match="not found"):
        service.set_active_key(404)

    with Session(bind=get_database().backend.engine) as session:
        row = session.get(HFAccessKey, key_id)
    assert row is not None
    assert row.is_active is True

###############################################################################
def test_set_active_key_raises_not_found_for_unknown_key(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()
    with pytest.raises(HFAccessKeyNotFoundError, match="not found"):
        service.set_active_key(404)
