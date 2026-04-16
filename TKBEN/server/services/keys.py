from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from TKBEN.server.common.utils.encryption import SymmetricCipher, get_hf_key_cipher
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.repositories.database.backend import database
from TKBEN.server.repositories.schemas.models import HFAccessKey


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
    def _session(self) -> Session:
        return Session(bind=database.backend.engine)

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

    def list_keys(self) -> list[dict[str, Any]]:
        stmt = select(HFAccessKey).order_by(
            HFAccessKey.created_at.desc(),
            HFAccessKey.id.desc(),
        )
        with self._session() as session:
            rows = session.execute(stmt).scalars().all()
        return [
            {
                "id": int(row.id),
                "created_at": row.created_at,
                "is_active": bool(row.is_active),
                "masked_preview": self.mask_key_preview(str(row.key_value or "")),
            }
            for row in rows
        ]

    # -------------------------------------------------------------------------
    def add_key(self, raw_key: str) -> dict[str, Any]:
        normalized_key = self.normalize_raw_key(raw_key)
        created_at = datetime.now(timezone.utc)

        encrypted_value = self.cipher.encrypt(normalized_key)
        with self._session() as session:
            rows = session.execute(select(HFAccessKey.key_value)).all()
            for (stored_value,) in rows:
                if not stored_value:
                    continue
                stored_text = str(stored_value)
                try:
                    decrypted_value = self.cipher.decrypt(stored_text)
                except ValueError:
                    logger.warning(
                        "Skipping undecryptable Hugging Face key while checking duplicates."
                    )
                    continue
                if decrypted_value == normalized_key:
                    raise HFAccessKeyConflictError(
                        "This Hugging Face key is already stored."
                    )
            key_row = HFAccessKey(
                key_value=encrypted_value,
                created_at=created_at,
                is_active=False,
            )
            try:
                session.add(key_row)
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise HFAccessKeyConflictError(
                    "This Hugging Face key is already stored."
                ) from exc
            session.refresh(key_row)

        if key_row.id is None:
            raise RuntimeError("Failed to save Hugging Face key.")

        return {
            "id": int(key_row.id),
            "created_at": key_row.created_at or created_at,
            "is_active": False,
            "masked_preview": self.mask_key_preview(encrypted_value),
        }

    # -------------------------------------------------------------------------
    def get_encrypted_key(self, key_id: int) -> str:
        with self._session() as session:
            row = session.execute(
                select(HFAccessKey).where(HFAccessKey.id == int(key_id)).limit(1)
            ).scalar_one_or_none()
        if row is None:
            raise HFAccessKeyNotFoundError("Hugging Face key not found.")

        encrypted_value = row.key_value
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

        with self._session() as session:
            row = session.execute(
                select(HFAccessKey).where(HFAccessKey.id == int(key_id)).limit(1)
            ).scalar_one_or_none()
            if row is None:
                raise HFAccessKeyNotFoundError("Hugging Face key not found.")
            if bool(row.is_active):
                raise HFAccessKeyValidationError(
                    "The active Hugging Face key cannot be deleted."
                )
            session.delete(row)
            session.commit()

    # -------------------------------------------------------------------------
    def set_active_key(self, key_id: int) -> None:
        with self._session() as session:
            row = session.execute(
                select(HFAccessKey).where(HFAccessKey.id == int(key_id)).limit(1)
            ).scalar_one_or_none()
            if row is None:
                raise HFAccessKeyNotFoundError("Hugging Face key not found.")
            # Keep activation idempotent: repeated activate calls must leave the key active.
            session.execute(
                update(HFAccessKey)
                .where(HFAccessKey.id != int(key_id))
                .values(is_active=False)
            )
            session.execute(
                update(HFAccessKey)
                .where(HFAccessKey.id == int(key_id))
                .values(is_active=True)
            )
            session.commit()

    # -------------------------------------------------------------------------
    def clear_active_key(self, key_id: int) -> None:
        with self._session() as session:
            row = session.execute(
                select(HFAccessKey.id).where(HFAccessKey.id == int(key_id)).limit(1)
            ).scalar_one_or_none()
            if row is None:
                raise HFAccessKeyNotFoundError("Hugging Face key not found.")
            session.execute(
                update(HFAccessKey)
                .where(HFAccessKey.id == int(key_id))
                .values(is_active=False)
            )
            session.commit()

    # -------------------------------------------------------------------------
    def get_active_key(self) -> str:
        with self._session() as session:
            row = session.execute(
                select(HFAccessKey)
                .where(HFAccessKey.is_active.is_(True))
                .order_by(HFAccessKey.id.desc())
                .limit(1)
            ).scalar_one_or_none()
        if row is None:
            raise HFAccessKeyValidationError(
                "No active Hugging Face access key is configured."
            )
        encrypted_value = row.key_value
        if not encrypted_value:
            raise HFAccessKeyValidationError(
                "No active Hugging Face access key is configured."
            )
        encrypted_text = str(encrypted_value)
        try:
            return self.cipher.decrypt(encrypted_text)
        except ValueError as exc:
            raise HFAccessKeyValidationError(
                self.get_decryption_error_message()
            ) from exc
