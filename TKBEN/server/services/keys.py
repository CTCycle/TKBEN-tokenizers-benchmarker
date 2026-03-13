from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.exc import IntegrityError

from TKBEN.server.common.utils.encryption import SymmetricCipher, get_hf_key_cipher
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.repositories.database.backend import database
from TKBEN.server.repositories.queries import keys as key_queries


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
        self._cipher: SymmetricCipher | None = None

    # -------------------------------------------------------------------------
    @property
    def cipher(self) -> SymmetricCipher:
        if self._cipher is None:
            self._cipher = get_hf_key_cipher()
        return self._cipher

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
    def get_decryption_error_message(self) -> str:
        return (
            "Stored Hugging Face key cannot be decrypted. "
            "Set a valid active key again using the current HF_KEYS_ENCRYPTION_KEY."
        )

    # -------------------------------------------------------------------------
    def is_legacy_plaintext_key(self, key_value: str) -> bool:
        normalized = key_value.strip()
        if not normalized.startswith("hf_"):
            return False
        if len(normalized) < 10:
            return False
        return " " not in normalized

    # -------------------------------------------------------------------------
    def migrate_plaintext_key(self, key_id: int, raw_key: str) -> None:
        encrypted_value = self.cipher.encrypt(raw_key)
        with database.backend.engine.begin() as conn:
            conn.execute(
                key_queries.UPDATE_HF_ACCESS_KEY_VALUE_BY_ID,
                {"key_value": encrypted_value, "key_id": key_id},
            )

    # -------------------------------------------------------------------------
    def list_keys(self) -> list[dict[str, Any]]:
        with database.backend.engine.connect() as conn:
            rows = conn.execute(key_queries.SELECT_HF_ACCESS_KEYS_FOR_LIST).fetchall()

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

        encrypted_value = self.cipher.encrypt(normalized_key)
        with database.backend.engine.begin() as conn:
            rows = conn.execute(key_queries.SELECT_HF_KEY_VALUES).fetchall()
            for row in rows:
                stored_value = self.read_row_value(row, "key_value", 0)
                if not stored_value:
                    continue
                stored_text = str(stored_value)
                try:
                    decrypted_value = self.cipher.decrypt(stored_text)
                except ValueError:
                    if self.is_legacy_plaintext_key(stored_text):
                        if stored_text.strip() == normalized_key:
                            raise HFAccessKeyConflictError(
                                "This Hugging Face key is already stored."
                            )
                        continue
                    logger.warning(
                        "Skipping undecryptable Hugging Face key while checking duplicates."
                    )
                    continue
                if decrypted_value == normalized_key:
                    raise HFAccessKeyConflictError(
                        "This Hugging Face key is already stored."
                    )
            try:
                conn.execute(
                    key_queries.INSERT_HF_ACCESS_KEY,
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
                key_queries.SELECT_HF_ACCESS_KEY_BY_VALUE,
                {"key_value": encrypted_value},
            ).first()

        if inserted is None:
            raise RuntimeError("Failed to save Hugging Face key.")

        key_id = int(self.read_row_value(inserted, "id", 0))
        inserted_created_at = (
            self.read_row_value(inserted, "created_at", 1) or created_at
        )
        return {
            "id": key_id,
            "created_at": inserted_created_at,
            "is_active": False,
            "masked_preview": self.mask_key_preview(encrypted_value),
        }

    # -------------------------------------------------------------------------
    def get_encrypted_key(self, key_id: int) -> str:
        with database.backend.engine.connect() as conn:
            row = conn.execute(
                key_queries.SELECT_HF_KEY_VALUE_BY_ID, {"key_id": key_id}
            ).first()
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
            raise HFAccessKeyValidationError("Deletion requires explicit confirmation.")

        with database.backend.engine.begin() as conn:
            row = conn.execute(
                key_queries.SELECT_HF_ACCESS_KEY_IS_ACTIVE_BY_ID,
                {"key_id": key_id},
            ).first()
            if row is None:
                raise HFAccessKeyNotFoundError("Hugging Face key not found.")
            is_active = bool(self.read_row_value(row, "is_active", 0))
            if is_active:
                raise HFAccessKeyValidationError(
                    "The active Hugging Face key cannot be deleted."
                )
            conn.execute(key_queries.DELETE_HF_ACCESS_KEY_BY_ID, {"key_id": key_id})

    # -------------------------------------------------------------------------
    def set_active_key(self, key_id: int) -> None:
        with database.backend.engine.begin() as conn:
            row = conn.execute(
                key_queries.SELECT_HF_ACCESS_KEY_ID_AND_IS_ACTIVE_BY_ID,
                {"key_id": key_id},
            ).first()
            if row is None:
                raise HFAccessKeyNotFoundError("Hugging Face key not found.")
            # Keep activation idempotent: repeated activate calls must leave the key active.
            conn.execute(
                key_queries.UPDATE_HF_ACCESS_KEYS_SET_ACTIVE_EXCEPT_ID,
                {"is_active": False, "key_id": key_id},
            )
            conn.execute(
                key_queries.UPDATE_HF_ACCESS_KEY_SET_ACTIVE_BY_ID,
                {"is_active": True, "key_id": key_id},
            )

    # -------------------------------------------------------------------------
    def clear_active_key(self, key_id: int) -> None:
        with database.backend.engine.begin() as conn:
            row = conn.execute(
                key_queries.SELECT_HF_ACCESS_KEY_ID_BY_ID, {"key_id": key_id}
            ).first()
            if row is None:
                raise HFAccessKeyNotFoundError("Hugging Face key not found.")
            conn.execute(
                key_queries.UPDATE_HF_ACCESS_KEY_SET_ACTIVE_BY_ID,
                {"is_active": False, "key_id": key_id},
            )

    # -------------------------------------------------------------------------
    def get_active_key(self) -> str:
        with database.backend.engine.connect() as conn:
            row = conn.execute(
                key_queries.SELECT_ACTIVE_HF_ACCESS_KEY, {"is_active": True}
            ).first()
        if row is None:
            raise HFAccessKeyValidationError(
                "No active Hugging Face access key is configured."
            )
        key_id = self.read_row_value(row, "id", 0)
        encrypted_value = self.read_row_value(row, "key_value", 1)
        if not encrypted_value:
            raise HFAccessKeyValidationError(
                "No active Hugging Face access key is configured."
            )
        encrypted_text = str(encrypted_value)
        try:
            return self.cipher.decrypt(encrypted_text)
        except ValueError as exc:
            if self.is_legacy_plaintext_key(encrypted_text):
                logger.warning(
                    "Active Hugging Face key is stored as plaintext legacy format; "
                    "migrating to encrypted storage."
                )
                try:
                    normalized_key_id = int(key_id)
                except (TypeError, ValueError):
                    normalized_key_id = None
                if normalized_key_id is not None:
                    try:
                        self.migrate_plaintext_key(
                            normalized_key_id, encrypted_text.strip()
                        )
                    except Exception:
                        logger.warning(
                            "Failed to migrate plaintext Hugging Face key for id=%s",
                            key_id,
                            exc_info=True,
                        )
                return encrypted_text.strip()
            raise HFAccessKeyValidationError(
                self.get_decryption_error_message()
            ) from exc
