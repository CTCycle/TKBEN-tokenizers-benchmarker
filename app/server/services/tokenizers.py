from __future__ import annotations

import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import HfApi
from tokenizers import Tokenizer as FastTokenizer
from transformers import AutoTokenizer

from server.configurations import get_server_settings
from server.common.utils.logger import logger
from server.contracts.tokenizers import (
    SupportedTokenizerPipeline,
    TokenizerDiscoveryItem,
    TokenizerDiscoveryQuery,
    TokenizerDiscoveryResponse,
)
from server.repositories.tokenizers import TokenizerRepository
from server.services.benchmarks import BenchmarkTools
from server.services.keys import HFAccessKeyService
from server.services.tokenizer_storage import TokenizerStorageMixin


###############################################################################
class TokenizerDownloadTimeoutError(TimeoutError):
    """Raised when a tokenizer provider load exceeds the job timeout."""


###############################################################################
class TokenizersService(TokenizerStorageMixin):
    """
    Service for fetching tokenizer discovery and catalog information from HuggingFace.
    """

    SUPPORTED_PIPELINE_TAGS = tuple(item.value for item in SupportedTokenizerPipeline)
    TOKENIZER_METADATA_FILES = frozenset({"config.json", "tokenizer_config.json"})
    FAST_TOKENIZER_FILES = frozenset({"tokenizer.json"})
    SENTENCEPIECE_TOKENIZER_FILES = frozenset(
        {
            "sentencepiece.bpe.model",
            "spiece.model",
            "tokenizer.model",
        }
    )
    TOKENIZER_DOWNLOAD_TIMEOUT_SECONDS = 120.0

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.repository = TokenizerRepository()
        self.key_service = HFAccessKeyService()
        self.benchmark_tools = BenchmarkTools()

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
        if not is_compatible:
            logger.warning(
                "Custom tokenizer %s is not compatible (source=%s)",
                tokenizer_name,
                normalized_filename,
            )
        else:
            previous_artifact = self.persist_custom_tokenizer_artifact(
                tokenizer_name, file_content
            )
            try:
                self.repository.upsert_tokenizer_source(
                    tokenizer_name,
                    source="custom",
                )
            except Exception:
                self.restore_custom_tokenizer_artifact(
                    tokenizer_name, previous_artifact
                )
                raise
            logger.info(
                "Persisted custom tokenizer: %s (source=%s)",
                tokenizer_name,
                normalized_filename,
            )

        return {
            "status": "success",
            "tokenizer_name": tokenizer_name,
            "is_compatible": is_compatible,
        }

    def has_available_tokenizer(self, tokenizer_id: str) -> bool:
        """Return whether a tokenizer can be loaded for reports or benchmarks."""
        source = self.repository.get_tokenizer_source(tokenizer_id)
        if source is None:
            return False
        if source == "custom":
            return self.has_custom_tokenizer_artifact(tokenizer_id)
        return self.has_cached_tokenizer(tokenizer_id)

    # -------------------------------------------------------------------------
    def remove_tokenizer(self, tokenizer_id: str) -> bool:
        tokenizer_name = self.validate_tokenizer_identifier(tokenizer_id)
        cache_dir = Path(self.get_tokenizer_cache_dir(tokenizer_name))
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
            except OSError as exc:
                logger.warning(
                    "Failed to remove tokenizer cache for %s; keeping database row: %s",
                    tokenizer_name,
                    exc,
                )
                raise RuntimeError(
                    f"Failed to remove tokenizer files for '{tokenizer_name}'."
                ) from exc

        # Commit the database deletion only after filesystem cleanup succeeds.
        return self.repository.delete_tokenizer(tokenizer_name)

    # -------------------------------------------------------------------------
    def list_tokenizer_catalog(
        self,
        search: str | None = None,
        source: Literal["all", "huggingface", "custom"] = "all",
        vocabulary_size_operator: Literal["at_least", "at_most"] = "at_least",
        vocabulary_size: int | None = None,
    ) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for (
            name,
            tokenizer_source,
            has_report,
            metadata,
        ) in self.repository.list_downloaded_tokenizer_catalog():
            has_artifact = (
                self.has_custom_tokenizer_artifact(name)
                if tokenizer_source == "custom"
                else self.has_cached_tokenizer(name)
            )
            if not has_artifact:
                continue
            parsed_size = None
            if isinstance(metadata, dict):
                value = metadata.get("vocabulary_size")
                if isinstance(value, int) and value >= 0:
                    parsed_size = value
            if parsed_size is None and tokenizer_source == "custom":
                try:
                    parsed_size = int(
                        FastTokenizer.from_file(
                            str(self.custom_tokenizer_artifact_path(name))
                        ).get_vocab_size()
                    )
                except Exception:  # noqa: BLE001
                    parsed_size = None
            catalog.append(
                {
                    "tokenizer_name": name,
                    "source": tokenizer_source,
                    "has_report": has_report,
                    "vocabulary_size": parsed_size,
                }
            )

        search_term = (search or "").strip().casefold()
        filtered = [
            item
            for item in catalog
            if (source == "all" or item["source"] == source)
            and (
                not search_term or search_term in str(item["tokenizer_name"]).casefold()
            )
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
    def discover_tokenizers(
        self,
        query: TokenizerDiscoveryQuery,
    ) -> TokenizerDiscoveryResponse:
        """Discover bounded Hugging Face repositories with tokenizer artifacts."""
        settings = get_server_settings().tokenizers
        needs_vocabulary_metadata = (
            query.vocabulary_size is not None or query.vocabulary_sort != "none"
        )
        candidate_limit = min(
            settings.max_discovery_candidates,
            query.limit * settings.metadata_candidate_multiplier,
        )

        hf_access_token = self.key_service.get_active_key()
        api = HfApi(token=hf_access_token)
        provider_kwargs: dict[str, Any] = {
            "sort": query.sort.value,
            "direction": -1,
            "limit": candidate_limit,
        }
        if query.search:
            provider_kwargs["search"] = query.search
        if query.author:
            provider_kwargs["author"] = query.author
        if query.pipeline_tag is not None:
            provider_kwargs["pipeline_tag"] = query.pipeline_tag.value
        if query.include_tags:
            provider_kwargs["filter"] = list(query.include_tags)
        if query.access != "all":
            provider_kwargs["gated"] = query.access == "gated"
        provider_kwargs["expand"] = [
            "siblings",
            "pipeline_tag",
            "library_name",
            "downloads",
            "likes",
            "lastModified",
            "gated",
            "tags",
        ]
        if needs_vocabulary_metadata:
            provider_kwargs["expand"].append("config")

        models = list(api.list_models(**provider_kwargs))
        items: list[TokenizerDiscoveryItem] = []
        for model in models:
            if not self._has_usable_tokenizer_artifacts(model):
                continue
            item = self._build_discovery_item(model)
            if not item.identifier:
                continue
            if query.pipeline_tag is None:
                if item.pipeline_tag not in self.SUPPORTED_PIPELINE_TAGS:
                    continue
            elif item.pipeline_tag != query.pipeline_tag.value:
                continue
            if query.exclude_tags and self._has_any_tag(item.tags, query.exclude_tags):
                continue
            if not self._matches_vocabulary(item, query):
                continue
            items.append(item)

        if query.vocabulary_sort != "none":
            items = self._sort_by_vocabulary(items, query.vocabulary_sort)

        return TokenizerDiscoveryResponse(
            items=items[: query.limit],
            count=min(len(items), query.limit),
            fetched_count=len(models),
        )

    # -------------------------------------------------------------------------
    @classmethod
    def _has_usable_tokenizer_artifacts(cls, model: Any) -> bool:
        """Return whether expanded Hub metadata exposes a root tokenizer resource.

        Discovery intentionally inspects repository metadata only. The download
        workflow remains responsible for loading the selected repository with
        ``AutoTokenizer`` and cleaning up an incompatible download.
        """
        siblings = getattr(model, "siblings", None)
        if siblings is None and isinstance(model, dict):
            siblings = model.get("siblings")
        if not siblings:
            return False

        root_files: set[str] = set()
        for sibling in siblings:
            filename = getattr(sibling, "rfilename", None)
            if filename is None and isinstance(sibling, dict):
                filename = sibling.get("rfilename") or sibling.get("path")
            if not isinstance(filename, str):
                continue
            normalized = filename.replace("\\", "/").strip("/")
            if not normalized or "/" in normalized:
                continue
            root_files.add(normalized.casefold())

        has_metadata = bool(root_files & cls.TOKENIZER_METADATA_FILES)
        has_fast_tokenizer = bool(root_files & cls.FAST_TOKENIZER_FILES)
        has_sentencepiece_tokenizer = bool(
            root_files & cls.SENTENCEPIECE_TOKENIZER_FILES
        )
        has_bpe_tokenizer = {"vocab.json", "merges.txt"}.issubset(root_files)
        has_wordpiece_tokenizer = "vocab.txt" in root_files

        return (
            (has_fast_tokenizer and has_metadata)
            or (has_sentencepiece_tokenizer and has_metadata)
            or has_bpe_tokenizer
            or (has_wordpiece_tokenizer and has_metadata)
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_discovery_item(model: Any) -> TokenizerDiscoveryItem:
        identifier = getattr(model, "modelId", None) or getattr(model, "id", None)
        identifier = identifier if isinstance(identifier, str) else ""
        pipeline_tag = getattr(model, "pipeline_tag", None)
        pipeline_tag = pipeline_tag if isinstance(pipeline_tag, str) else None
        library_name = getattr(model, "library_name", None)
        library_name = library_name if isinstance(library_name, str) else None
        last_modified = getattr(model, "lastModified", None)
        if last_modified is None:
            last_modified = getattr(model, "last_modified", None)
        if isinstance(last_modified, datetime):
            last_modified = last_modified.isoformat()
        elif last_modified is not None and not isinstance(last_modified, str):
            last_modified = str(last_modified)
        tags = getattr(model, "tags", None)
        normalized_tags = [tag for tag in tags or [] if isinstance(tag, str)]
        config = getattr(model, "config", None)
        vocabulary_size = None
        if isinstance(config, dict):
            configured_size = config.get("vocab_size")
            if (
                isinstance(configured_size, int)
                and not isinstance(configured_size, bool)
                and configured_size >= 0
            ):
                vocabulary_size = configured_size
        return TokenizerDiscoveryItem(
            identifier=identifier,
            pipeline_tag=pipeline_tag,
            library_name=library_name,
            downloads=TokenizersService._non_negative_int(
                getattr(model, "downloads", None)
            ),
            likes=TokenizersService._non_negative_int(getattr(model, "likes", None)),
            last_modified=last_modified,
            gated=getattr(model, "gated", None),
            tags=normalized_tags,
            vocabulary_size=vocabulary_size,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _non_negative_int(value: object) -> int | None:
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else None
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _has_any_tag(tags: list[str], excluded: list[str]) -> bool:
        known = {tag.casefold() for tag in tags}
        return any(tag.casefold() in known for tag in excluded)

    # -------------------------------------------------------------------------
    @staticmethod
    def _matches_vocabulary(
        item: TokenizerDiscoveryItem,
        query: TokenizerDiscoveryQuery,
    ) -> bool:
        if query.vocabulary_size is None:
            return True
        if item.vocabulary_size is None:
            return False
        operator = query.vocabulary_operator or "at_least"
        return (
            item.vocabulary_size >= query.vocabulary_size
            if operator == "at_least"
            else item.vocabulary_size <= query.vocabulary_size
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _sort_by_vocabulary(
        items: list[TokenizerDiscoveryItem],
        ordering: Literal["ascending", "descending"],
    ) -> list[TokenizerDiscoveryItem]:
        known = [item for item in items if item.vocabulary_size is not None]
        unknown = [item for item in items if item.vocabulary_size is None]
        known.sort(
            key=lambda item: item.vocabulary_size or 0,
            reverse=ordering == "descending",
        )
        return known + unknown

    # -------------------------------------------------------------------------
    @staticmethod
    def _cleanup_tokenizer_cache_after_timeout(
        worker_thread: threading.Thread,
        cache_dir: Path,
    ) -> None:
        worker_thread.join()
        shutil.rmtree(cache_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    def _load_tokenizer_with_timeout(
        self,
        tokenizer_id: str,
        cache_dir: str,
        hf_access_token: str | None,
    ) -> Any:
        result_holder: dict[str, Any] = {}
        error_holder: dict[str, BaseException] = {}

        def load() -> None:
            try:
                result_holder["tokenizer"] = AutoTokenizer.from_pretrained(
                    tokenizer_id,
                    cache_dir=cache_dir,
                    token=hf_access_token,
                )
            except BaseException as exc:  # noqa: BLE001
                error_holder["error"] = exc

        worker_thread = threading.Thread(target=load, daemon=True)
        worker_thread.start()
        worker_thread.join(timeout=self.TOKENIZER_DOWNLOAD_TIMEOUT_SECONDS)
        if worker_thread.is_alive():
            threading.Thread(
                target=self._cleanup_tokenizer_cache_after_timeout,
                args=(worker_thread, Path(cache_dir)),
                daemon=True,
            ).start()
            raise TokenizerDownloadTimeoutError(
                "Tokenizer download timed out after "
                f"{self.TOKENIZER_DOWNLOAD_TIMEOUT_SECONDS:.1f} seconds."
            )

        error = error_holder.get("error")
        if error is not None:
            raise error
        tokenizer = result_holder.get("tokenizer")
        if tokenizer is None:
            raise RuntimeError("Tokenizer download produced no tokenizer result.")
        return tokenizer

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
                is_persisted = self.repository.tokenizer_exists(tokenizer_id)
                source = (
                    self.repository.get_tokenizer_source(tokenizer_id)
                    if is_persisted
                    else None
                )
                has_required_artifact = (
                    self.has_custom_tokenizer_artifact(tokenizer_id)
                    if source == "custom"
                    else self.has_cached_tokenizer(tokenizer_id)
                )
                if is_persisted and has_required_artifact:
                    already_downloaded.append(tokenizer_id)
                else:
                    if source == "custom":
                        raise ValueError(
                            f"Custom tokenizer '{tokenizer_id}' is missing its canonical artifact."
                        )
                    cache_dir = self.get_tokenizer_cache_dir(tokenizer_id)
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    self._load_tokenizer_with_timeout(
                        tokenizer_id,
                        cache_dir,
                        hf_access_token,
                    )
                    # Keep cached tokenizer files because benchmark runs load
                    # tokenizers locally with local_files_only=True.
                    self.repository.insert_if_missing(
                        tokenizer_id,
                        source="huggingface",
                    )
                    downloaded.append(tokenizer_id)
            except Exception as exc:  # noqa: BLE001
                cache_dir = Path(self.get_tokenizer_cache_dir(tokenizer_id))
                if not isinstance(exc, TokenizerDownloadTimeoutError):
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
