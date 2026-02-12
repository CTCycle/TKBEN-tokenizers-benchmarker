from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from TKBEN.server.common.constants import HF_KEYS_ENCRYPTION_KEY


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
    return SymmetricCipher(HF_KEYS_ENCRYPTION_KEY)
