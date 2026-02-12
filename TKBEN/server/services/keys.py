from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import sqlalchemy
from sqlalchemy.exc import IntegrityError

from TKBEN.server.common.utils.encryption import get_hf_key_cipher
from TKBEN.server.repositories.database.backend import database


###############################################################################
class HFAccessKeyError(Exception):
    pass


###############################################################################
class HFAccessKeyValidationError(HFAccessKeyError):
    pass


###############################################################################
class HFAccessKeyConflictError(HFAccessKeyError):
    pass


###############################################################################
class HFAccessKeyNotFoundError(HFAccessKeyError):
    pass


###############################################################################
class HFAccessKeyService:
    def __init__(self) -> None:
        self.cipher = get_hf_key_cipher()

    # -------------------------------------------------------------------------
    def normalize_raw_key(self, raw_key: str) -> str:
        normalized = raw_key.strip() if isinstance(raw_key, str) else ""
        if not normalized:
            raise HFAccessKeyValidationError("Hugging Face key cannot be empty.")
        return normalized

    # -------------------------------------------------------------------------
    def mask_key_preview(self, encrypted_key: str) -> str:
        return "*" * len(encrypted_key)

    # -------------------------------------------------------------------------
    def mask_key_full(self, raw_key: str) -> str:
        return "*" * len(raw_key)

    # -------------------------------------------------------------------------
    def read_row_value(self, row: Any, key: str, index: int) -> Any:
        if hasattr(row, "_mapping"):
            return row._mapping.get(key)
        if isinstance(row, tuple) and len(row) > index:
            return row[index]
        return None

    # -------------------------------------------------------------------------
    def list_keys(self) -> list[dict[str, Any]]:
        query = sqlalchemy.text(
            'SELECT "id", "key_value", "created_at", "is_active" '
            'FROM "hf_access_keys" ORDER BY "created_at" DESC, "id" DESC'
        )
        with database.backend.engine.connect() as conn:
            rows = conn.execute(query).fetchall()

        keys: list[dict[str, Any]] = []
        for row in rows:
            key_id = int(self.read_row_value(row, "id", 0))
            encrypted_value = str(self.read_row_value(row, "key_value", 1) or "")
            created_at = self.read_row_value(row, "created_at", 2)
            is_active = bool(self.read_row_value(row, "is_active", 3))
            keys.append(
                {
                    "id": key_id,
                    "created_at": created_at,
                    "is_active": is_active,
                    "masked_preview": self.mask_key_preview(encrypted_value),
                }
            )
        return keys

    # -------------------------------------------------------------------------
    def add_key(self, raw_key: str) -> dict[str, Any]:
        normalized_key = self.normalize_raw_key(raw_key)
        created_at = datetime.now(timezone.utc)

        list_query = sqlalchemy.text('SELECT "key_value" FROM "hf_access_keys"')
        insert_query = sqlalchemy.text(
            'INSERT INTO "hf_access_keys" ("key_value", "created_at", "is_active") '
            'VALUES (:key_value, :created_at, :is_active)'
        )
        select_query = sqlalchemy.text(
            'SELECT "id", "created_at", "is_active" '
            'FROM "hf_access_keys" WHERE "key_value" = :key_value LIMIT 1'
        )

        encrypted_value = self.cipher.encrypt(normalized_key)
        with database.backend.engine.begin() as conn:
            rows = conn.execute(list_query).fetchall()
            for row in rows:
                stored_value = self.read_row_value(row, "key_value", 0)
                if not stored_value:
                    continue
                if self.cipher.decrypt(str(stored_value)) == normalized_key:
                    raise HFAccessKeyConflictError(
                        "This Hugging Face key is already stored."
                    )
            try:
                conn.execute(
                    insert_query,
                    {
                        "key_value": encrypted_value,
                        "created_at": created_at,
                        "is_active": False,
                    },
                )
            except IntegrityError as exc:
                raise HFAccessKeyConflictError(
                    "This Hugging Face key is already stored."
                ) from exc
            inserted = conn.execute(
                select_query,
                {"key_value": encrypted_value},
            ).first()

        if inserted is None:
            raise RuntimeError("Failed to save Hugging Face key.")

        key_id = int(self.read_row_value(inserted, "id", 0))
        inserted_created_at = self.read_row_value(inserted, "created_at", 1) or created_at
        return {
            "id": key_id,
            "created_at": inserted_created_at,
            "is_active": False,
            "masked_preview": self.mask_key_preview(encrypted_value),
        }

    # -------------------------------------------------------------------------
    def get_encrypted_key(self, key_id: int) -> str:
        query = sqlalchemy.text(
            'SELECT "key_value" FROM "hf_access_keys" WHERE "id" = :key_id LIMIT 1'
        )
        with database.backend.engine.connect() as conn:
            row = conn.execute(query, {"key_id": key_id}).first()
        if row is None:
            raise HFAccessKeyNotFoundError("Hugging Face key not found.")

        encrypted_value = self.read_row_value(row, "key_value", 0)
        if not encrypted_value:
            raise HFAccessKeyNotFoundError("Hugging Face key not found.")
        return str(encrypted_value)

    # -------------------------------------------------------------------------
    def get_masked_key(self, key_id: int) -> str:
        return self.mask_key_full(self.get_encrypted_key(key_id))

    # -------------------------------------------------------------------------
    def get_revealed_key(self, key_id: int) -> str:
        return self.get_encrypted_key(key_id)

    # -------------------------------------------------------------------------
    def delete_key(self, key_id: int, confirm: bool) -> None:
        if not confirm:
            raise HFAccessKeyValidationError(
                "Deletion requires explicit confirmation."
            )

        find_query = sqlalchemy.text(
            'SELECT "is_active" FROM "hf_access_keys" WHERE "id" = :key_id LIMIT 1'
        )
        delete_query = sqlalchemy.text(
            'DELETE FROM "hf_access_keys" WHERE "id" = :key_id'
        )

        with database.backend.engine.begin() as conn:
            row = conn.execute(find_query, {"key_id": key_id}).first()
            if row is None:
                raise HFAccessKeyNotFoundError("Hugging Face key not found.")
            is_active = bool(self.read_row_value(row, "is_active", 0))
            if is_active:
                raise HFAccessKeyValidationError(
                    "The active Hugging Face key cannot be deleted."
                )
            conn.execute(delete_query, {"key_id": key_id})

    # -------------------------------------------------------------------------
    def set_active_key(self, key_id: int) -> None:
        find_query = sqlalchemy.text(
            'SELECT "id", "is_active" FROM "hf_access_keys" WHERE "id" = :key_id LIMIT 1'
        )
        clear_query = sqlalchemy.text(
            'UPDATE "hf_access_keys" SET "is_active" = :is_active'
        )
        activate_query = sqlalchemy.text(
            'UPDATE "hf_access_keys" SET "is_active" = :is_active WHERE "id" = :key_id'
        )

        with database.backend.engine.begin() as conn:
            row = conn.execute(find_query, {"key_id": key_id}).first()
            if row is None:
                raise HFAccessKeyNotFoundError("Hugging Face key not found.")
            is_currently_active = bool(self.read_row_value(row, "is_active", 1))
            conn.execute(clear_query, {"is_active": False})
            if not is_currently_active:
                conn.execute(activate_query, {"is_active": True, "key_id": key_id})

    # -------------------------------------------------------------------------
    def clear_active_key(self, key_id: int) -> None:
        find_query = sqlalchemy.text(
            'SELECT "id" FROM "hf_access_keys" WHERE "id" = :key_id LIMIT 1'
        )
        clear_query = sqlalchemy.text(
            'UPDATE "hf_access_keys" SET "is_active" = :is_active WHERE "id" = :key_id'
        )
        with database.backend.engine.begin() as conn:
            row = conn.execute(find_query, {"key_id": key_id}).first()
            if row is None:
                raise HFAccessKeyNotFoundError("Hugging Face key not found.")
            conn.execute(clear_query, {"is_active": False, "key_id": key_id})

    # -------------------------------------------------------------------------
    def get_active_key(self) -> str:
        query = sqlalchemy.text(
            'SELECT "key_value" FROM "hf_access_keys" '
            'WHERE "is_active" = :is_active ORDER BY "id" DESC LIMIT 1'
        )
        with database.backend.engine.connect() as conn:
            row = conn.execute(query, {"is_active": True}).first()
        if row is None:
            raise HFAccessKeyValidationError(
                "No active Hugging Face access key is configured."
            )
        encrypted_value = self.read_row_value(row, "key_value", 0)
        if not encrypted_value:
            raise HFAccessKeyValidationError(
                "No active Hugging Face access key is configured."
            )
        return self.cipher.decrypt(str(encrypted_value))
