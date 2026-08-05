from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sized
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import HfApi
from tokenizers import Tokenizer as FastTokenizer
from transformers import AutoTokenizer

from server.common.utils.logger import logger
from server.repositories.tokenizers import TokenizerRepository
from server.services.benchmarks import BenchmarkTools
from server.services.custom_tokenizers import get_custom_tokenizer_registry
from server.services.keys import HFAccessKeyService
from server.services.tokenizer_storage import TokenizerStorageMixin

###############################################################################
class TokenizersService(TokenizerStorageMixin):
    """
    Service for fetching tokenizer information from HuggingFace.

    Service for tokenizer scanning and metadata retrieval
    from HuggingFace.
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

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.repository = TokenizerRepository()
        self.key_service = HFAccessKeyService()
        self.benchmark_tools = BenchmarkTools()
        self.custom_tokenizer_registry = get_custom_tokenizer_registry()

    # -------------------------------------------------------------------------
    def register_custom_tokenizer_from_upload(
        self,
        file_content: bytes,
        normalized_filename: str,
        safe_stem: str,
    ) -> dict[str, Any]:
        if not file_content:
            raise ValueError("Uploaded file is empty.")

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".json",
                delete=False,
            ) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            tokenizer = FastTokenizer.from_file(temp_path)
        except Exception as exc:
            logger.warning("Failed to load tokenizer from uploaded file: %s", exc)
            raise ValueError(f"Failed to load tokenizer: {exc}") from exc
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink()
                except FileNotFoundError:
                    pass

        is_compatible = self.benchmark_tools.is_tokenizer_compatible(tokenizer)
        tokenizer_name = f"CUSTOM_{safe_stem}"
        if is_compatible:
            self.custom_tokenizer_registry.set(tokenizer_name, tokenizer)
            logger.info(
                "Loaded custom tokenizer: %s (source=%s)",
                tokenizer_name,
                normalized_filename,
            )
        else:
            logger.warning(
                "Custom tokenizer %s is not compatible (source=%s)",
                tokenizer_name,
                normalized_filename,
            )

        return {
            "status": "success",
            "tokenizer_name": tokenizer_name,
            "is_compatible": is_compatible,
        }

    # -------------------------------------------------------------------------
    def clear_custom_tokenizers(self) -> None:
        self.custom_tokenizer_registry.clear()

    # -------------------------------------------------------------------------
    def is_tokenizer_persisted(self, tokenizer_id: str) -> bool:
        return self.repository.tokenizer_exists(tokenizer_id)

    # -------------------------------------------------------------------------
    def insert_tokenizer_if_missing(self, tokenizer_id: str) -> None:
        self.repository.insert_if_missing(tokenizer_id)

    # -------------------------------------------------------------------------
    def list_downloaded_tokenizers(self) -> list[str]:
        names: list[str] = []
        for name in self.repository.list_downloaded_tokenizers():
            if self.has_cached_tokenizer(name):
                names.append(name)
        return names

    # -------------------------------------------------------------------------
    def remove_downloaded_tokenizer(self, tokenizer_id: str) -> bool:
        tokenizer_name = self.validate_tokenizer_identifier(tokenizer_id)
        removed = self.repository.delete_tokenizer(tokenizer_name)
        if not removed:
            return False
        cache_dir = Path(self.get_tokenizer_cache_dir(tokenizer_name))
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        return True

    # -------------------------------------------------------------------------
    def list_tokenizer_catalog(
        self,
        search: str | None = None,
        source: Literal["all", "huggingface", "custom"] = "all",
        vocabulary_size_operator: Literal["at_least", "at_most"] = "at_least",
        vocabulary_size: int | None = None,
    ) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for name, has_report, metadata in self.repository.list_downloaded_tokenizer_catalog():
            if not self.has_cached_tokenizer(name):
                continue
            parsed_size = None
            if isinstance(metadata, dict):
                value = metadata.get("vocabulary_size")
                if isinstance(value, int) and value >= 0:
                    parsed_size = value
            catalog.append({
                "tokenizer_name": name,
                "source": "huggingface",
                "has_report": has_report,
                "vocabulary_size": parsed_size,
            })

        for name, tokenizer in self.custom_tokenizer_registry.snapshot().items():
            parsed_size: int | None = None
            get_size = getattr(tokenizer, "get_vocab_size", None)
            if callable(get_size):
                try:
                    value = get_size()
                    if isinstance(value, int) and value >= 0:
                        parsed_size = value
                except Exception:  # noqa: BLE001
                    pass
            if parsed_size is None:
                get_vocab = getattr(tokenizer, "get_vocab", None)
                if callable(get_vocab):
                    try:
                        vocabulary = get_vocab()
                        if isinstance(vocabulary, Sized):
                            parsed_size = len(vocabulary)
                    except Exception:  # noqa: BLE001
                        pass
            catalog.append({
                "tokenizer_name": name,
                "source": "custom",
                "has_report": False,
                "vocabulary_size": parsed_size,
            })

        search_term = (search or "").strip().casefold()
        filtered = [
            item for item in catalog
            if (source == "all" or item["source"] == source)
            and (not search_term or search_term in str(item["tokenizer_name"]).casefold())
            and (
                vocabulary_size is None
                or (
                    item["vocabulary_size"] is not None
                    and (
                        item["vocabulary_size"] >= vocabulary_size
                        if vocabulary_size_operator == "at_least"
                        else item["vocabulary_size"] <= vocabulary_size
                    )
                )
            )
        ]
        return sorted(filtered, key=lambda item: str(item["tokenizer_name"]).casefold())

    # -------------------------------------------------------------------------
    def get_tokenizer_identifiers(self, limit: int = 100) -> list[Any]:
        """
        Retrieve the most downloaded tokenizer identifiers from Hugging Face.

        Args:
            limit: Maximum number of identifiers to request (default 100).

        Returns:
            List with the identifiers of the retrieved tokenizers ordered by
            popularity (downloads).
        """
        hf_access_token = self.key_service.get_active_key()
        api = HfApi(token=hf_access_token)

        models = api.list_models(
            search="tokenizer", sort="downloads", direction=-1, limit=limit
        )

        identifiers = [
            model_id
            for model in models
            if isinstance(model_id := getattr(model, "modelId", None), str)
            and getattr(model, "pipeline_tag", None) in self.PIPELINE_TAGS
        ]

        return identifiers

    # -------------------------------------------------------------------------
    def download_and_persist(
        self,
        tokenizers: list[str],
        progress_callback: Any | None = None,
        should_stop: Any | None = None,
    ) -> dict[str, Any]:
        requested = self.normalize_tokenizer_identifiers(tokenizers)
        hf_access_token = self.key_service.get_active_key()
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
                "failed_details": [],
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
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    AutoTokenizer.from_pretrained(
                        tokenizer_id,
                        cache_dir=cache_dir,
                        token=hf_access_token,
                    )
                    # Keep cached tokenizer files because benchmark runs load
                    # tokenizers locally with local_files_only=True.
                    self.insert_tokenizer_if_missing(tokenizer_id)
                    downloaded.append(tokenizer_id)
            except Exception as exc:  # noqa: BLE001
                cache_dir = Path(self.get_tokenizer_cache_dir(tokenizer_id))
                shutil.rmtree(cache_dir, ignore_errors=True)
                reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:240]}"
                logger.warning(
                    "Failed to download tokenizer %s (%s)", tokenizer_id, reason
                )
                logger.debug(
                    "Tokenizer download failed for %s", tokenizer_id, exc_info=True
                )
                failed.append(f"{tokenizer_id}: {reason}")

            if callable(progress_callback):
                progress_callback(((index + 1) / total) * 100.0)

        return {
            "status": "success",
            "downloaded": downloaded,
            "already_downloaded": already_downloaded,
            "failed": [item.split(": ", 1)[0] for item in failed],
            "failed_details": failed,
            "requested_count": len(requested),
            "downloaded_count": len(downloaded),
            "already_downloaded_count": len(already_downloaded),
            "failed_count": len(failed),
        }

    # -------------------------------------------------------------------------
