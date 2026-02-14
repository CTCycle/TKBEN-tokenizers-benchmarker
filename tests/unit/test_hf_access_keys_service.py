from __future__ import annotations

from datetime import datetime, timezone

import pytest

from TKBEN.server.repositories.database.backend import database
from TKBEN.server.services.keys import (
    HFAccessKeyConflictError,
    HFAccessKeyNotFoundError,
    HFAccessKeyService,
    HFAccessKeyValidationError,
)


###############################################################################
class FakeResult:
    def __init__(self, rows: list[tuple[str]] | None = None, row: tuple | None = None) -> None:
        self.rows = rows or []
        self.row = row

    # -------------------------------------------------------------------------
    def fetchall(self) -> list[tuple[str]]:
        return self.rows

    # -------------------------------------------------------------------------
    def first(self) -> tuple | None:
        return self.row


###############################################################################
def test_get_active_key_raises_validation_error_on_invalid_decryption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = HFAccessKeyService()

    class FailingCipher:
        def decrypt(self, encrypted_value: str) -> str:
            raise ValueError("invalid token")

    class ActiveKeyConnection:
        def __enter__(self) -> "ActiveKeyConnection":
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> bool:
            return False

        def execute(self, statement, params=None) -> FakeResult:  # noqa: ANN001
            return FakeResult(row=(1, "encrypted-value"))

    class ActiveKeyEngine:
        def connect(self) -> ActiveKeyConnection:
            return ActiveKeyConnection()

    service._cipher = FailingCipher()  # type: ignore[assignment]
    monkeypatch.setattr(database.backend, "engine", ActiveKeyEngine())

    with pytest.raises(HFAccessKeyValidationError, match="cannot be decrypted"):
        service.get_active_key()


###############################################################################
def test_add_key_skips_undecryptable_rows_during_duplicate_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = HFAccessKeyService()
    inserted_at = datetime.now(timezone.utc)

    class RecoveringCipher:
        def encrypt(self, plaintext: str) -> str:
            return f"enc:{plaintext}"

        def decrypt(self, encrypted_value: str) -> str:
            if encrypted_value == "stale-encrypted":
                raise ValueError("invalid token")
            if encrypted_value.startswith("enc:"):
                return encrypted_value[4:]
            return encrypted_value

    class TransactionConnection:
        def __init__(self) -> None:
            self.insert_payloads: list[dict[str, object]] = []

        def __enter__(self) -> "TransactionConnection":
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> bool:
            return False

        def execute(self, statement, params=None) -> FakeResult:  # noqa: ANN001
            sql = str(statement)
            if (
                'SELECT "key_value" FROM "hf_access_keys"' in sql
                and 'WHERE "key_value"' not in sql
            ):
                return FakeResult(rows=[("stale-encrypted",)])
            if 'INSERT INTO "hf_access_keys"' in sql:
                self.insert_payloads.append(dict(params or {}))
                return FakeResult()
            if 'SELECT "id", "created_at", "is_active"' in sql:
                return FakeResult(row=(7, inserted_at, False))
            raise AssertionError(f"Unexpected SQL query: {sql}")

    class BeginEngine:
        def __init__(self) -> None:
            self.tx = TransactionConnection()

        def begin(self) -> TransactionConnection:
            return self.tx

    fake_engine = BeginEngine()
    service._cipher = RecoveringCipher()  # type: ignore[assignment]
    monkeypatch.setattr(database.backend, "engine", fake_engine)

    result = service.add_key("hf_test_key")

    assert result["id"] == 7
    assert result["is_active"] is False
    assert fake_engine.tx.insert_payloads
    assert fake_engine.tx.insert_payloads[0]["key_value"] == "enc:hf_test_key"


###############################################################################
def test_get_active_key_migrates_legacy_plaintext_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = HFAccessKeyService()
    update_calls: list[dict[str, object]] = []

    class LegacyAwareCipher:
        def encrypt(self, plaintext: str) -> str:
            return f"enc:{plaintext}"

        def decrypt(self, encrypted_value: str) -> str:
            if encrypted_value.startswith("enc:"):
                return encrypted_value[4:]
            raise ValueError("invalid token")

    class SelectConnection:
        def __enter__(self) -> "SelectConnection":
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> bool:
            return False

        def execute(self, statement, params=None) -> FakeResult:  # noqa: ANN001
            return FakeResult(row=(3, "hf_legacy_token_123"))

    class UpdateConnection:
        def __enter__(self) -> "UpdateConnection":
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> bool:
            return False

        def execute(self, statement, params=None) -> FakeResult:  # noqa: ANN001
            update_calls.append(dict(params or {}))
            return FakeResult()

    class MixedEngine:
        def connect(self) -> SelectConnection:
            return SelectConnection()

        def begin(self) -> UpdateConnection:
            return UpdateConnection()

    service._cipher = LegacyAwareCipher()  # type: ignore[assignment]
    monkeypatch.setattr(database.backend, "engine", MixedEngine())

    key = service.get_active_key()

    assert key == "hf_legacy_token_123"
    assert update_calls
    assert update_calls[0]["key_id"] == 3
    assert update_calls[0]["key_value"] == "enc:hf_legacy_token_123"


###############################################################################
def test_add_key_detects_duplicate_for_legacy_plaintext_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = HFAccessKeyService()

    class LegacyAwareCipher:
        def encrypt(self, plaintext: str) -> str:
            return f"enc:{plaintext}"

        def decrypt(self, encrypted_value: str) -> str:
            if encrypted_value.startswith("enc:"):
                return encrypted_value[4:]
            raise ValueError("invalid token")

    class SelectOnlyConnection:
        def __enter__(self) -> "SelectOnlyConnection":
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> bool:
            return False

        def execute(self, statement, params=None) -> FakeResult:  # noqa: ANN001
            sql = str(statement)
            if 'SELECT "key_value" FROM "hf_access_keys"' in sql:
                return FakeResult(rows=[("hf_legacy_token_123",)])
            raise AssertionError(f"Unexpected SQL query: {sql}")

    class SelectOnlyEngine:
        def begin(self) -> SelectOnlyConnection:
            return SelectOnlyConnection()

    service._cipher = LegacyAwareCipher()  # type: ignore[assignment]
    monkeypatch.setattr(database.backend, "engine", SelectOnlyEngine())

    with pytest.raises(HFAccessKeyConflictError) as exc_info:
        service.add_key("hf_legacy_token_123")

    assert "already stored" in str(exc_info.value)


###############################################################################
def test_set_active_key_is_idempotent_for_already_active_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = HFAccessKeyService()
    execute_calls: list[tuple[str, dict[str, object]]] = []

    class ActivationConnection:
        def __enter__(self) -> "ActivationConnection":
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> bool:
            return False

        def execute(self, statement, params=None) -> FakeResult:  # noqa: ANN001
            sql = str(statement)
            payload = dict(params or {})
            execute_calls.append((sql, payload))
            if 'SELECT "id", "is_active" FROM "hf_access_keys"' in sql:
                return FakeResult(row=(11, True))
            return FakeResult()

    class ActivationEngine:
        def begin(self) -> ActivationConnection:
            return ActivationConnection()

    monkeypatch.setattr(database.backend, "engine", ActivationEngine())

    service.set_active_key(11)

    assert any(
        'UPDATE "hf_access_keys" SET "is_active" = :is_active WHERE "id" != :key_id' in sql
        and payload == {"is_active": False, "key_id": 11}
        for sql, payload in execute_calls
    )
    assert any(
        'UPDATE "hf_access_keys" SET "is_active" = :is_active WHERE "id" = :key_id' in sql
        and payload == {"is_active": True, "key_id": 11}
        for sql, payload in execute_calls
    )


###############################################################################
def test_set_active_key_raises_not_found_for_unknown_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = HFAccessKeyService()

    class MissingConnection:
        def __enter__(self) -> "MissingConnection":
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> bool:
            return False

        def execute(self, statement, params=None) -> FakeResult:  # noqa: ANN001
            return FakeResult(row=None)

    class MissingEngine:
        def begin(self) -> MissingConnection:
            return MissingConnection()

    monkeypatch.setattr(database.backend, "engine", MissingEngine())

    with pytest.raises(HFAccessKeyNotFoundError, match="not found"):
        service.set_active_key(404)
