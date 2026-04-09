from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from TKBEN.server.configurations import get_app_settings


###############################################################################
class SymmetricCipher:
    def __init__(self, key_value: str) -> None:
        try:
            self.fernet = Fernet(key_value.encode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                "HF_KEYS_ENCRYPTION_KEY must be a valid Fernet key."
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
def get_hf_key_cipher() -> SymmetricCipher:
    app_settings = get_app_settings()
    key_value = app_settings.hf_keys_encryption_key
    if not key_value:
        raise RuntimeError(
            "HF_KEYS_ENCRYPTION_KEY must be configured in the environment."
        )
    return SymmetricCipher(key_value)
