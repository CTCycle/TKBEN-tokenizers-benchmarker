from __future__ import annotations

import os

from TKBEN.server.common.constants import TOKENIZERS_PATH
from TKBEN.server.common.utils.security import (
    ensure_path_is_within,
    normalize_identifier,
)


###############################################################################
class TokenizerStorageMixin:
    TOKENIZER_ID_MAX_LENGTH = 160

    # -------------------------------------------------------------------------
    def validate_tokenizer_identifier(self, value: str) -> str:
        return normalize_identifier(
            value,
            "Tokenizer identifier",
            max_length=self.TOKENIZER_ID_MAX_LENGTH,
        )

    # -------------------------------------------------------------------------
    def normalize_tokenizer_identifiers(self, tokenizers: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        invalid: list[str] = []
        for value in tokenizers:
            name = str(value).strip()
            if not name:
                continue
            try:
                safe_name = self.validate_tokenizer_identifier(name)
            except ValueError:
                invalid.append(name)
                continue
            if safe_name in seen:
                continue
            seen.add(safe_name)
            normalized.append(safe_name)
        if invalid:
            preview = ", ".join(invalid[:3])
            if len(invalid) > 3:
                preview = f"{preview}, ..."
            raise ValueError(f"Invalid tokenizer identifier(s): {preview}")
        return normalized

    # -------------------------------------------------------------------------
    def get_tokenizer_cache_dir(self, tokenizer_id: str) -> str:
        safe_id = self.validate_tokenizer_identifier(tokenizer_id)
        safe_name = safe_id.replace("/", "__")
        candidate = os.path.join(TOKENIZERS_PATH, safe_name)
        return ensure_path_is_within(TOKENIZERS_PATH, candidate)

    # -------------------------------------------------------------------------
    def has_cached_tokenizer(self, tokenizer_id: str) -> bool:
        cache_dir = self.get_tokenizer_cache_dir(tokenizer_id)
        if not os.path.isdir(cache_dir):
            return False
        for _, _, files in os.walk(cache_dir):
            if files:
                return True
        return False

    # -------------------------------------------------------------------------
    def build_huggingface_url(self, tokenizer_name: str) -> str | None:
        normalized = str(tokenizer_name).strip()
        if not normalized:
            return None
        if normalized.upper().startswith("CUSTOM_"):
            return None
        if " " in normalized:
            return None
        return f"https://huggingface.co/{normalized}"
