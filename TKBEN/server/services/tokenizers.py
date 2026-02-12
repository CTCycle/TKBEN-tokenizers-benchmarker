from __future__ import annotations

import os
from typing import Any

from huggingface_hub import HfApi
import sqlalchemy
from transformers import AutoTokenizer

from TKBEN.server.common.constants import TOKENIZERS_PATH
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.repositories.database.backend import database


###############################################################################
class TokenizersService:
    """
    Service for fetching tokenizer information from HuggingFace.

    This is a webapp-specific service that provides tokenizer scanning
    functionality without the desktop app dependencies.
    """

    PIPELINE_TAGS = [
        "text-generation",
        "fill-mask",
        "text-classification",
        "token-classification",
        "text2text-generation",
        "question-answering",
        "sentence-similarity",
        "translation",
        "summarization",
        "conversational",
        "zero-shot-classification",
    ]

    def __init__(self, hf_access_token: str | None = None) -> None:
        self.hf_access_token = hf_access_token

    # -------------------------------------------------------------------------
    def normalize_tokenizer_identifiers(self, tokenizers: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in tokenizers:
            name = str(value).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            normalized.append(name)
        return normalized

    # -------------------------------------------------------------------------
    def get_tokenizer_cache_dir(self, tokenizer_id: str) -> str:
        safe_name = tokenizer_id.replace("/", "__")
        return os.path.join(TOKENIZERS_PATH, safe_name)

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
    def is_tokenizer_persisted(self, tokenizer_id: str) -> bool:
        query = sqlalchemy.text(
            'SELECT 1 FROM "tokenizer" WHERE "name" = :name LIMIT 1'
        )
        with database.backend.engine.connect() as conn:
            row = conn.execute(query, {"name": tokenizer_id}).first()
        return row is not None

    # -------------------------------------------------------------------------
    def insert_tokenizer_if_missing(self, tokenizer_id: str) -> None:
        query = sqlalchemy.text(
            'INSERT INTO "tokenizer" ("name") '
            'SELECT :name '
            'WHERE NOT EXISTS (SELECT 1 FROM "tokenizer" WHERE "name" = :name)'
        )
        with database.backend.engine.begin() as conn:
            conn.execute(query, {"name": tokenizer_id})

    # -------------------------------------------------------------------------
    def list_downloaded_tokenizers(self) -> list[str]:
        query = sqlalchemy.text(
            'SELECT "name" FROM "tokenizer" ORDER BY "name" ASC'
        )
        with database.backend.engine.connect() as conn:
            rows = conn.execute(query).fetchall()
        names: list[str] = []
        for row in rows:
            if hasattr(row, "_mapping"):
                name = str(row._mapping["name"])
            else:
                name = str(row[0])
            if self.has_cached_tokenizer(name):
                names.append(name)
        return names

    def get_tokenizer_identifiers(self, limit: int = 100) -> list[Any]:
        """
        Retrieve the most downloaded tokenizer identifiers from Hugging Face.

        Args:
            limit: Maximum number of identifiers to request (default 100).

        Returns:
            List with the identifiers of the retrieved tokenizers ordered by
            popularity (downloads).
        """
        api = HfApi(token=self.hf_access_token) if self.hf_access_token else HfApi()

        try:
            models = api.list_models(
                search="tokenizer", sort="downloads", direction=-1, limit=limit
            )
        except Exception:
            logger.warning("Failed to retrieve tokenizer identifiers from HuggingFace")
            logger.debug("Tokenizer identifier fetch failed", exc_info=True)
            return []

        identifiers = [m.modelId for m in models]

        return identifiers

    # -------------------------------------------------------------------------
    def download_and_persist(
        self,
        tokenizers: list[str],
        progress_callback: Any | None = None,
        should_stop: Any | None = None,
    ) -> dict[str, Any]:
        requested = self.normalize_tokenizer_identifiers(tokenizers)
        downloaded: list[str] = []
        already_downloaded: list[str] = []
        failed: list[str] = []

        total = len(requested)
        if total == 0:
            return {
                "status": "success",
                "downloaded": downloaded,
                "already_downloaded": already_downloaded,
                "failed": failed,
                "requested_count": 0,
                "downloaded_count": 0,
                "already_downloaded_count": 0,
                "failed_count": 0,
            }

        for index, tokenizer_id in enumerate(requested):
            if callable(should_stop) and should_stop():
                break

            try:
                is_persisted = self.is_tokenizer_persisted(tokenizer_id)
                has_cached = self.has_cached_tokenizer(tokenizer_id)
                if is_persisted and has_cached:
                    already_downloaded.append(tokenizer_id)
                else:
                    cache_dir = self.get_tokenizer_cache_dir(tokenizer_id)
                    os.makedirs(cache_dir, exist_ok=True)
                    AutoTokenizer.from_pretrained(
                        tokenizer_id,
                        cache_dir=cache_dir,
                        token=self.hf_access_token,
                    )
                    self.insert_tokenizer_if_missing(tokenizer_id)
                    downloaded.append(tokenizer_id)
            except Exception:
                logger.warning("Failed to download tokenizer %s", tokenizer_id)
                logger.debug(
                    "Tokenizer download failed for %s", tokenizer_id, exc_info=True
                )
                failed.append(tokenizer_id)

            if callable(progress_callback):
                progress_callback(((index + 1) / total) * 100.0)

        return {
            "status": "success",
            "downloaded": downloaded,
            "already_downloaded": already_downloaded,
            "failed": failed,
            "requested_count": len(requested),
            "downloaded_count": len(downloaded),
            "already_downloaded_count": len(already_downloaded),
            "failed_count": len(failed),
        }

