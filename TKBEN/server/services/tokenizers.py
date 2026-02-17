from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from huggingface_hub import HfApi, ModelCard
import sqlalchemy
from transformers import AutoTokenizer

from TKBEN.server.common.constants import TOKENIZERS_PATH
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.configurations import server_settings
from TKBEN.server.repositories.database.backend import database
from TKBEN.server.repositories.serialization.data import TokenizerReportSerializer
from TKBEN.server.services.keys import HFAccessKeyService, HFAccessKeyValidationError


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

    REPORT_VERSION = 1

    def __init__(self) -> None:
        self.key_service = HFAccessKeyService()
        self.report_serializer = TokenizerReportSerializer()
        self.histogram_bins = max(5, int(server_settings.datasets.histogram_bins))
        self.special_token_pattern = re.compile(
            r"^(?:\[.*\]|<.*>|\{.*\}|</?s>|</?pad>|UNK|PAD)$",
            re.IGNORECASE,
        )

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
            'VALUES (:name) '
            'ON CONFLICT ("name") DO NOTHING'
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
    def get_hf_access_token_for_metadata(self) -> str | None:
        try:
            return self.key_service.get_active_key()
        except HFAccessKeyValidationError:
            logger.warning(
                "No decryptable active Hugging Face key found. "
                "Proceeding with anonymous tokenizer metadata lookup."
            )
            return None

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
                        token=hf_access_token,
                    )
                    # Keep cached tokenizer files because benchmark runs load
                    # tokenizers locally with local_files_only=True.
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

    # -------------------------------------------------------------------------
    def extract_model_card_summary(
        self,
        card_data: Any,
        card_content: str | None = None,
    ) -> str | None:
        candidate_keys = (
            "description",
            "model_description",
            "summary",
            "model_summary",
        )
        payload = card_data
        if payload is not None and hasattr(payload, "to_dict") and callable(payload.to_dict):
            try:
                payload = payload.to_dict()
            except Exception:
                payload = card_data
        if isinstance(payload, dict):
            for key in candidate_keys:
                value = payload.get(key)
                if isinstance(value, str):
                    trimmed = value.strip()
                    if trimmed:
                        return trimmed

        content = str(card_content or "").strip()
        if not content:
            return None

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) == 3:
                content = parts[2].strip()

        lines = [line.strip() for line in content.splitlines()]
        paragraph_lines: list[str] = []
        started = False
        for line in lines:
            if not line:
                if started:
                    break
                continue
            if line.startswith("#"):
                continue
            paragraph_lines.append(line)
            started = True
        if not paragraph_lines:
            return None
        summary = " ".join(paragraph_lines).strip()
        return summary or None

    # -------------------------------------------------------------------------
    def resolve_hf_repo_metadata(
        self,
        tokenizer_name: str,
    ) -> tuple[str | None, str | None]:
        normalized = str(tokenizer_name).strip()
        canonical_url = self.build_huggingface_url(normalized)
        if canonical_url is None:
            return None, None

        hf_access_token = self.get_hf_access_token_for_metadata()
        api = HfApi(token=hf_access_token)
        description: str | None = None

        try:
            model_info = api.model_info(
                normalized,
                expand=["cardData"],
                token=hf_access_token,
            )
            model_id = getattr(model_info, "id", None)
            if isinstance(model_id, str) and model_id.strip():
                canonical_url = self.build_huggingface_url(model_id) or canonical_url
            card_data = getattr(model_info, "card_data", None)
            description = self.extract_model_card_summary(card_data=card_data)
        except Exception:
            logger.debug(
                "Failed to retrieve model info for tokenizer %s",
                normalized,
                exc_info=True,
            )

        if description is None:
            try:
                model_card = ModelCard.load(normalized, token=hf_access_token)
                card_content = getattr(model_card, "content", "")
                description = self.extract_model_card_summary(
                    card_data=None,
                    card_content=card_content,
                )
            except Exception:
                logger.debug(
                    "Failed to load model card for tokenizer %s",
                    normalized,
                    exc_info=True,
                )

        return description, canonical_url

    # -------------------------------------------------------------------------
    def find_cached_file(
        self,
        cache_dir: str,
        candidate_names: tuple[str, ...],
    ) -> str | None:
        candidate_set = {name.lower() for name in candidate_names}
        for root, dirs, files in os.walk(cache_dir):
            dirs.sort()
            files_sorted = sorted(files)
            for filename in files_sorted:
                if filename.lower() in candidate_set:
                    return os.path.join(root, filename)
        return None

    # -------------------------------------------------------------------------
    def load_json_if_present(self, path: str | None) -> dict[str, Any]:
        if path is None:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            if isinstance(payload, dict):
                return payload
            return {}
        except Exception:
            logger.debug("Failed to parse tokenizer metadata file: %s", path, exc_info=True)
            return {}

    # -------------------------------------------------------------------------
    def detect_algorithm_type(
        self,
        tokenizer: Any,
        tokenizer_json: dict[str, Any],
        tokenizer_config: dict[str, Any],
    ) -> str | None:
        model_block = tokenizer_json.get("model")
        if isinstance(model_block, dict):
            model_type = model_block.get("type")
            if isinstance(model_type, str) and model_type.strip():
                return model_type.strip()

        backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
        backend_model = getattr(backend_tokenizer, "model", None)
        backend_class_name = getattr(backend_model, "__class__", type(None)).__name__
        if isinstance(backend_class_name, str) and backend_class_name not in {"", "NoneType"}:
            return backend_class_name

        tokenizer_class = tokenizer_config.get("tokenizer_class")
        if isinstance(tokenizer_class, str):
            lowered = tokenizer_class.lower()
            if "wordpiece" in lowered:
                return "WordPiece"
            if "bpe" in lowered:
                return "BPE"
            if "unigram" in lowered or "sentencepiece" in lowered:
                return "Unigram"
            if "wordlevel" in lowered:
                return "WordLevel"

        return None

    # -------------------------------------------------------------------------
    def normalize_special_tokens(self, tokenizer: Any) -> list[str]:
        token_map = getattr(tokenizer, "special_tokens_map", {})
        normalized: list[str] = []
        if not isinstance(token_map, dict):
            return normalized
        for value in token_map.values():
            if isinstance(value, str):
                normalized.append(value)
            elif isinstance(value, list):
                normalized.extend(str(item) for item in value if item is not None)
        return sorted({token for token in normalized if token})

    # -------------------------------------------------------------------------
    def resolve_casing_hint(
        self,
        tokenizer: Any,
        tokenizer_config: dict[str, Any],
    ) -> bool | None:
        do_lower_case = getattr(tokenizer, "do_lower_case", None)
        if isinstance(do_lower_case, bool):
            return do_lower_case
        init_kwargs = getattr(tokenizer, "init_kwargs", {})
        if isinstance(init_kwargs, dict):
            value = init_kwargs.get("do_lower_case")
            if isinstance(value, bool):
                return value
        config_value = tokenizer_config.get("do_lower_case")
        if isinstance(config_value, bool):
            return config_value
        return None

    # -------------------------------------------------------------------------
    def resolve_normalization_hint(
        self,
        tokenizer_json: dict[str, Any],
        tokenizer_config: dict[str, Any],
    ) -> str | None:
        normalizer_block = tokenizer_json.get("normalizer")
        if isinstance(normalizer_block, dict):
            normalizer_type = normalizer_block.get("type")
            if isinstance(normalizer_type, str) and normalizer_type.strip():
                return normalizer_type.strip()
        normalizer_class = tokenizer_config.get("normalizer_class")
        if isinstance(normalizer_class, str) and normalizer_class.strip():
            return normalizer_class.strip()
        return None

    # -------------------------------------------------------------------------
    def resolve_description(
        self,
        tokenizer_config: dict[str, Any],
        model_config: dict[str, Any],
    ) -> str | None:
        for payload in (tokenizer_config, model_config):
            if not isinstance(payload, dict):
                continue
            description = payload.get("description")
            if isinstance(description, str):
                trimmed = description.strip()
                if trimmed:
                    return trimmed
        return None

    # -------------------------------------------------------------------------
    def compute_subword_word_stats(
        self,
        vocab_tokens: list[str],
        special_tokens: set[str],
    ) -> dict[str, Any]:
        special_lookup = {token for token in special_tokens if token}
        subword_count = 0
        word_count = 0

        for raw_token in vocab_tokens:
            token = str(raw_token).strip()
            if not token:
                continue
            if token in special_lookup or self.special_token_pattern.match(token):
                continue

            # Deterministic heuristic:
            # - "##" prefix indicates WordPiece continuation pieces.
            # - "▁" marks SentencePiece word starts; non-leading usage is treated as continuation.
            # - "Ġ" marks byte-level BPE word starts; non-leading usage is treated as continuation.
            is_subword = (
                token.startswith("##")
                or ("▁" in token and not token.startswith("▁"))
                or ("Ġ" in token and not token.startswith("Ġ"))
            )
            if is_subword:
                subword_count += 1
            else:
                word_count += 1

        considered_count = subword_count + word_count
        subword_percentage = (
            (float(subword_count) / float(considered_count)) * 100.0
            if considered_count > 0
            else 0.0
        )
        word_percentage = (
            (float(word_count) / float(considered_count)) * 100.0
            if considered_count > 0
            else 0.0
        )
        ratio = (
            float(subword_count) / float(word_count)
            if word_count > 0
            else None
        )

        return {
            "heuristic": "wordpiece_hashes_sentencepiece_underscore_bytebpe_G",
            "subword_count": int(subword_count),
            "word_count": int(word_count),
            "considered_count": int(considered_count),
            "subword_percentage": float(subword_percentage),
            "word_percentage": float(word_percentage),
            "subword_to_word_ratio": ratio,
        }

    # -------------------------------------------------------------------------
    def resolve_tokenizer_persistence_mode(
        self,
        tokenizer_name: str,
        cache_dir: str,
    ) -> dict[str, str]:
        _ = tokenizer_name
        _ = cache_dir
        # Downloaded tokenizer loading relies on AutoTokenizer.from_pretrained(..., local_files_only=True)
        # in benchmark/report paths, so filesystem artifacts are required today.
        # DB-only mode is intentionally not selected because there is no DB reconstruction path.
        return {
            "persistence_mode": "filesystem_required",
            "persistence_reason": (
                "Runtime tokenizer loading uses local_files_only=True with cached files; "
                "no DB-only reconstruction path exists."
            ),
        }

    # -------------------------------------------------------------------------
    def compute_histogram(self, lengths: list[int]) -> dict[str, Any]:
        if not lengths:
            return {
                "bins": [],
                "counts": [],
                "bin_edges": [],
                "min_length": 0,
                "max_length": 0,
                "mean_length": 0.0,
                "median_length": 0.0,
            }

        sorted_lengths = sorted(lengths)
        min_length = sorted_lengths[0]
        max_length = sorted_lengths[-1]
        bin_count = max(1, self.histogram_bins)
        span = (max_length - min_length) + 1
        bin_width = max(1, int(math.ceil(span / max(1, bin_count))))

        edges = [min_length]
        for _ in range(bin_count):
            edges.append(edges[-1] + bin_width)

        counts = [0] * (len(edges) - 1)
        for length in sorted_lengths:
            index = min((length - min_length) // bin_width, len(counts) - 1)
            counts[index] += 1

        bins: list[str] = []
        for idx in range(len(counts)):
            left = int(edges[idx])
            right = int(edges[idx + 1] - 1)
            if right < left:
                right = left
            bins.append(f"{left}-{right}" if left != right else f"{left}")

        total = len(sorted_lengths)
        midpoint = total // 2
        if total % 2 == 0:
            median_length = (sorted_lengths[midpoint - 1] + sorted_lengths[midpoint]) / 2.0
        else:
            median_length = float(sorted_lengths[midpoint])

        return {
            "bins": bins,
            "counts": counts,
            "bin_edges": [float(edge) for edge in edges],
            "min_length": int(min_length),
            "max_length": int(max_length),
            "mean_length": float(sum(sorted_lengths) / total),
            "median_length": float(median_length),
        }

    # -------------------------------------------------------------------------
    def extract_vocabulary(self, tokenizer: Any) -> list[tuple[int, str]]:
        vocab_func = getattr(tokenizer, "get_vocab", None)
        if not callable(vocab_func):
            return []
        raw_vocab = vocab_func()
        if not isinstance(raw_vocab, dict):
            return []

        pairs: list[tuple[int, str]] = []
        for token, token_id in raw_vocab.items():
            try:
                pairs.append((int(token_id), str(token)))
            except Exception:
                continue

        pairs.sort(key=lambda item: (item[0], item[1]))
        return pairs

    # -------------------------------------------------------------------------
    def generate_and_store_report(
        self,
        tokenizer_name: str,
        progress_callback: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        name = str(tokenizer_name).strip()
        if not name:
            raise ValueError("Tokenizer name must be provided.")

        cache_dir = self.get_tokenizer_cache_dir(name)
        if not self.has_cached_tokenizer(name):
            raise ValueError(
                f"Tokenizer '{name}' is not downloaded. Download it before validation."
            )

        if progress_callback:
            progress_callback(5.0)

        tokenizer = AutoTokenizer.from_pretrained(
            name,
            cache_dir=cache_dir,
            local_files_only=True,
        )

        if callable(should_stop) and should_stop():
            return {}

        if progress_callback:
            progress_callback(20.0)

        tokenizer_json_path = self.find_cached_file(cache_dir, ("tokenizer.json",))
        tokenizer_config_path = self.find_cached_file(cache_dir, ("tokenizer_config.json",))
        model_config_path = self.find_cached_file(cache_dir, ("config.json",))

        tokenizer_json = self.load_json_if_present(tokenizer_json_path)
        tokenizer_config = self.load_json_if_present(tokenizer_config_path)
        model_config = self.load_json_if_present(model_config_path)

        vocab_pairs = self.extract_vocabulary(tokenizer)
        vocabulary_tokens = [token for _, token in vocab_pairs]
        vocabulary_rows = [
            {
                "token_id": token_id,
                "vocabulary_tokens": token,
                "decoded_tokens": token,
            }
            for token_id, token in vocab_pairs
        ]
        lengths = [len(token) for _, token in vocab_pairs]

        if callable(should_stop) and should_stop():
            return {}

        if progress_callback:
            progress_callback(55.0)

        self.report_serializer.replace_tokenizer_vocabulary(name, vocabulary_rows)

        special_tokens = self.normalize_special_tokens(tokenizer)
        special_token_set = set(special_tokens)
        subword_word_stats = self.compute_subword_word_stats(
            vocab_tokens=vocabulary_tokens,
            special_tokens=special_token_set,
        )
        algorithm_type = self.detect_algorithm_type(
            tokenizer=tokenizer,
            tokenizer_json=tokenizer_json,
            tokenizer_config=tokenizer_config,
        )
        normalization_hint = self.resolve_normalization_hint(tokenizer_json, tokenizer_config)
        casing_hint = self.resolve_casing_hint(tokenizer, tokenizer_config)
        config_description = self.resolve_description(tokenizer_config, model_config)
        hf_description, huggingface_url = self.resolve_hf_repo_metadata(name)
        description = config_description or hf_description
        persistence_info = self.resolve_tokenizer_persistence_mode(name, cache_dir)
        histogram = self.compute_histogram(lengths)
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        global_stats = {
            "vocabulary_size": int(len(vocab_pairs)),
            "tokenizer_algorithm": algorithm_type,
            "tokenizer_class": getattr(tokenizer, "__class__", type(None)).__name__,
            "has_special_tokens": bool(special_tokens),
            "special_tokens": special_tokens,
            "special_tokens_count": int(len(special_tokens)),
            "do_lower_case": casing_hint,
            "normalization_hint": normalization_hint,
            "token_length_measure": "character_count",
            "subword_word_stats": subword_word_stats,
            "persistence_mode": persistence_info["persistence_mode"],
            "persistence_reason": persistence_info["persistence_reason"],
        }

        report_payload = {
            "report_version": self.REPORT_VERSION,
            "created_at": created_at,
            "tokenizer_name": name,
            "description": description,
            "huggingface_url": huggingface_url,
            "global_stats": global_stats,
            "token_length_histogram": histogram,
            "vocabulary_size": int(len(vocab_pairs)),
        }

        report_id = self.report_serializer.save_tokenizer_report(report_payload)
        report_payload["report_id"] = int(report_id)

        if progress_callback:
            progress_callback(100.0)

        return report_payload

    # -------------------------------------------------------------------------
    def get_latest_tokenizer_report(self, tokenizer_name: str) -> dict[str, Any] | None:
        return self.report_serializer.load_latest_tokenizer_report(tokenizer_name)

    # -------------------------------------------------------------------------
    def get_tokenizer_report_by_id(self, report_id: int) -> dict[str, Any] | None:
        return self.report_serializer.load_tokenizer_report_by_id(report_id)

    # -------------------------------------------------------------------------
    def get_tokenizer_report_vocabulary(
        self,
        report_id: int,
        offset: int,
        limit: int,
    ) -> dict[str, Any] | None:
        return self.report_serializer.load_tokenizer_vocabulary_page(
            report_id=report_id,
            offset=offset,
            limit=limit,
        )
