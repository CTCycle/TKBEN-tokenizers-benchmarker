from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from server.common.path import RESOURCES_PATH, ROOT_DIR

MATERIAL_FILE_ENV = "HF_KEYS_ENCRYPTION_MATERIAL_FILE"
MATERIAL_PURPOSE = "hugging_face_access_keys"

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
def _material_path() -> Path:
    configured = os.getenv(MATERIAL_FILE_ENV, "").strip()
    if not configured:
        return RESOURCES_PATH / "hf-key-material.json"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path

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
def _get_active_material(payload: dict[str, object]) -> str | None:
    purpose_store = payload.get(MATERIAL_PURPOSE)
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
def _ensure_material() -> str:
    path = _material_path()
    payload = _read_material_store(path)
    existing = _get_active_material(payload)
    if existing is not None:
        return existing

    now = datetime.now(UTC).replace(tzinfo=None).isoformat()
    payload[MATERIAL_PURPOSE] = {
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
    _write_material_store(path, payload)
    purpose_store = payload[MATERIAL_PURPOSE]
    assert isinstance(purpose_store, dict)
    versions = purpose_store["versions"]
    assert isinstance(versions, dict)
    record = versions["1"]
    assert isinstance(record, dict)
    return str(record["key_material"])

###############################################################################
def get_hf_key_cipher() -> SymmetricCipher:
    return SymmetricCipher(_ensure_material())
