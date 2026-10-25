# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from server.common.path import DATA_PATH

MATERIAL_PURPOSE = "hugging_face_access_keys"
DATABASE_PURPOSE = "database_connection_password"


###############################################################################
class SymmetricCipher:

    # -------------------------------------------------------------------------
    def __init__(self, key_value: str) -> None:
        try:
            self.fernet = Fernet(key_value.encode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                "The Hugging Face key material file contains an invalid Fernet key."
            ) from exc

    # -------------------------------------------------------------------------
    def encrypt(self, plaintext: str) -> str:
        return self.fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    # -------------------------------------------------------------------------
    def decrypt(self, encrypted_value: str) -> str:
        try:
            return self.fernet.decrypt(encrypted_value.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("Unable to decrypt stored Hugging Face key.") from exc


###############################################################################
class KeyMaterialNotInitializedError(RuntimeError):
    pass


###############################################################################
def _material_path() -> Path:
    return DATA_PATH / "hf-key-material.json"


###############################################################################
def _read_material_store(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Hugging Face key material file is invalid.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Hugging Face key material file must contain an object.")
    return payload


###############################################################################
def _write_material_store(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


###############################################################################
def _get_active_material(payload: dict[str, object], purpose: str) -> str | None:
    purpose_store = payload.get(purpose)
    if not isinstance(purpose_store, dict):
        return None
    active_version = purpose_store.get("active_version")
    versions = purpose_store.get("versions")
    if not isinstance(active_version, (int, str)) or not isinstance(versions, dict):
        return None
    record = versions.get(str(int(active_version)))
    if not isinstance(record, dict) or not bool(record.get("is_active", False)):
        return None
    key_material = record.get("key_material")
    return str(key_material) if isinstance(key_material, str) and key_material else None


###############################################################################
def get_purpose_cipher(path: Path, purpose: str) -> SymmetricCipher:
    """Return the active encryption key scoped to a purpose in the shared key store."""

    existing = _get_active_material(_read_material_store(path), purpose)
    if existing is None:
        raise KeyMaterialNotInitializedError(
            "Encryption key is not initialized. Generate it in Settings > Keys first."
        )
    return SymmetricCipher(existing)


###############################################################################
def is_key_material_initialized(path: Path | None = None) -> bool:
    payload = _read_material_store(path or _material_path())
    return all(
        _get_active_material(payload, purpose) is not None
        for purpose in (MATERIAL_PURPOSE, DATABASE_PURPOSE)
    )


###############################################################################
def initialize_key_material(path: Path | None = None) -> None:
    """Create missing key material per purpose; never replace an existing key."""

    target = path or _material_path()
    payload = _read_material_store(target)
    missing = [
        purpose
        for purpose in (MATERIAL_PURPOSE, DATABASE_PURPOSE)
        if _get_active_material(payload, purpose) is None
    ]
    if not missing:
        raise FileExistsError("Encryption key material already exists.")
    now = datetime.now(UTC).replace(tzinfo=None).isoformat()
    for purpose in missing:
        payload[purpose] = {
            "active_version": 1,
            "versions": {
                "1": {
                    "key_material": Fernet.generate_key().decode("utf-8"),
                    "is_active": True,
                    "seeded_at": now,
                    "activated_at": now,
                }
            },
        }
    _write_material_store(target, payload)


###############################################################################
def get_hf_key_cipher() -> SymmetricCipher:
    return get_purpose_cipher(_material_path(), MATERIAL_PURPOSE)
