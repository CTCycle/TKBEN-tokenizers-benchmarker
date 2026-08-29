from __future__ import annotations

import os
import tempfile
from pathlib import Path

from server.common.path import TOKENIZERS_PATH
from server.common.utils.security import (
    ensure_path_is_within,
    normalize_identifier,
)

###############################################################################
class TokenizerStorageMixin:
    TOKENIZER_ID_MAX_LENGTH = 160
    TOKENIZER_ARTIFACT_NAMES = (
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.json",
        "vocab.txt",
    )

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
        candidate = TOKENIZERS_PATH / safe_name
        return ensure_path_is_within(TOKENIZERS_PATH, candidate)

    # -------------------------------------------------------------------------
    def has_cached_tokenizer(self, tokenizer_id: str) -> bool:
        cache_dir = Path(self.get_tokenizer_cache_dir(tokenizer_id))
        if not cache_dir.is_dir():
            return False
        return any(
            path.is_file()
            and path.name in self.TOKENIZER_ARTIFACT_NAMES
            and path.stat().st_size > 0
            for path in cache_dir.rglob("*")
        )

    # -------------------------------------------------------------------------
    def custom_tokenizer_artifact_path(self, tokenizer_id: str) -> Path:
        cache_dir = Path(self.get_tokenizer_cache_dir(tokenizer_id))
        return Path(ensure_path_is_within(cache_dir, cache_dir / "tokenizer.json"))

    # -------------------------------------------------------------------------
    def has_custom_tokenizer_artifact(self, tokenizer_id: str) -> bool:
        artifact_path = self.custom_tokenizer_artifact_path(tokenizer_id)
        return artifact_path.is_file() and artifact_path.stat().st_size > 0

    # -------------------------------------------------------------------------
    def persist_custom_tokenizer_artifact(
        self, tokenizer_id: str, content: bytes
    ) -> bytes | None:
        if not content:
            raise ValueError("Custom tokenizer artifact is empty.")
        cache_dir = Path(self.get_tokenizer_cache_dir(tokenizer_id))
        cache_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.custom_tokenizer_artifact_path(tokenizer_id)
        previous_content = artifact_path.read_bytes() if artifact_path.exists() else None
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=cache_dir,
                prefix=".tokenizer.json.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = temporary_file.name
            os.replace(temporary_path, artifact_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    Path(temporary_path).unlink()
                except FileNotFoundError:
                    pass
        return previous_content

    # -------------------------------------------------------------------------
    def restore_custom_tokenizer_artifact(
        self, tokenizer_id: str, previous_content: bytes | None
    ) -> None:
        artifact_path = self.custom_tokenizer_artifact_path(tokenizer_id)
        if previous_content is None:
            try:
                artifact_path.unlink()
            except FileNotFoundError:
                pass
            return
        self.persist_custom_tokenizer_artifact(tokenizer_id, previous_content)

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
