from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError

from TKBEN.server.common.utils.encryption import SymmetricCipher, get_hf_key_cipher
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.repositories.hf_access_keys import HFAccessKeyRepository


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
    def __init__(self, repository: HFAccessKeyRepository | None = None) -> None:
        self._cipher: SymmetricCipher | None = None
        self.repository = repository or HFAccessKeyRepository()

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
    def get_decryption_error_message(self) -> str:
        return (
            "Stored Hugging Face key cannot be decrypted. "
            "Set a valid active key again using the current HF_KEYS_ENCRYPTION_KEY."
        )

    # -------------------------------------------------------------------------
    def list_keys(self) -> list[dict[str, object]]:
        rows = self.repository.list_all()
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
    def add_key(self, raw_key: str) -> dict[str, object]:
        normalized_key = self.normalize_raw_key(raw_key)
        created_at = datetime.now(timezone.utc)

        encrypted_value = self.cipher.encrypt(normalized_key)
        for stored_text in self.repository.list_encrypted_values():
            try:
                decrypted_value = self.cipher.decrypt(stored_text)
            except ValueError:
                logger.warning(
                    "Skipping undecryptable Hugging Face key while checking duplicates."
                )
                continue
            if decrypted_value == normalized_key:
                raise HFAccessKeyConflictError("This Hugging Face key is already stored.")

        try:
            key_row = self.repository.insert_key(encrypted_value, created_at)
        except IntegrityError as exc:
            raise HFAccessKeyConflictError(
                "This Hugging Face key is already stored."
            ) from exc

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
        row = self.repository.get_by_id(key_id)
        if row is None or not row.key_value:
            raise HFAccessKeyNotFoundError("Hugging Face key not found.")
        return str(row.key_value)

    # -------------------------------------------------------------------------
    def get_masked_key(self, key_id: int) -> str:
        return self.mask_key_full(self.get_encrypted_key(key_id))

    # -------------------------------------------------------------------------
    def get_revealed_key(self, key_id: int) -> str:
        encrypted_text = self.get_encrypted_key(key_id)
        try:
            return self.cipher.decrypt(encrypted_text)
        except ValueError as exc:
            raise HFAccessKeyValidationError(
                self.get_decryption_error_message()
            ) from exc

    # -------------------------------------------------------------------------
    def delete_key(self, key_id: int, confirm: bool) -> None:
        if not confirm:
            raise HFAccessKeyValidationError("Deletion requires explicit confirmation.")

        row = self.repository.get_by_id(key_id)
        if row is None:
            raise HFAccessKeyNotFoundError("Hugging Face key not found.")
        if bool(row.is_active):
            raise HFAccessKeyValidationError(
                "The active Hugging Face key cannot be deleted."
            )

        deleted = self.repository.delete_by_id(key_id)
        if not deleted:
            raise HFAccessKeyNotFoundError("Hugging Face key not found.")

    # -------------------------------------------------------------------------
    def set_active_key(self, key_id: int) -> None:
        activated = self.repository.activate(key_id)
        if not activated:
            raise HFAccessKeyNotFoundError("Hugging Face key not found.")

    # -------------------------------------------------------------------------
    def clear_active_key(self, key_id: int) -> None:
        deactivated = self.repository.deactivate(key_id)
        if not deactivated:
            raise HFAccessKeyNotFoundError("Hugging Face key not found.")

    # -------------------------------------------------------------------------
    def get_active_key(self) -> str:
        row = self.repository.get_active()
        if row is None or not row.key_value:
            raise HFAccessKeyValidationError(
                "No active Hugging Face access key is configured."
            )

        encrypted_text = str(row.key_value)
        try:
            return self.cipher.decrypt(encrypted_text)
        except ValueError as exc:
            raise HFAccessKeyValidationError(
                self.get_decryption_error_message()
            ) from exc
