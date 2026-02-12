from __future__ import annotations

import os
from typing import Any

import pandas as pd
import transformers
from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer

from TKBEN.server.common.constants import SOURCES_PATH, TOKENIZER_PATH
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.services.keys import HFAccessKeyService


# [DOWNLOADS]
###############################################################################
class DatasetManager:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.dataset = configuration.get("DATASET", {})
        self.dataset_corpus = self.dataset.get("corpus", "wikitext")
        self.dataset_config = self.dataset.get("config", "wikitext-103-v1")
        self.has_custom_dataset = configuration.get("use_custom_dataset", False)
        self.configuration = configuration
        self.key_service = HFAccessKeyService()

    # -------------------------------------------------------------------------
    def get_dataset_name(self) -> str | Any:
        if self.dataset_config:
            return f"{self.dataset_corpus}/{self.dataset_config}"
        return self.dataset_corpus

    # -------------------------------------------------------------------------
    def dataset_download(self) -> None | dict[str, Any]:
        """
        Load the dataset configured by the user, pulling either local CSV files or
        downloading an open dataset from Hugging Face.

        Keyword arguments:
        None

        Return value:
        Dictionary containing the loaded dataset objects keyed by name, or None
        when the operation fails or yields no data.
        """
        datasets = {}
        base_path = SOURCES_PATH
        # load a custom text dataset from .csv file
        if self.has_custom_dataset:
            csv_files = [
                os.path.join(base_path, fn)
                for fn in os.listdir(base_path)
                if fn.lower().endswith(".csv")
            ]
            if not csv_files:
                # do not return anything if no files are found and custom dataset is required
                logger.warning(
                    f"No CSV files found in custom dataset folder: {base_path}"
                )
                return None
            else:
                # if multiple files are found only the first one will be loaded
                if len(csv_files) > 1:
                    logger.warning(
                        f"Multiple CSV files found in {base_path}, using the first one: {os.path.basename(csv_files[0])}"
                    )
                file_path = csv_files[0]
                try:
                    df = pd.read_csv(file_path)
                except Exception:
                    logger.warning("Failed to load custom dataset from %s", file_path)
                    logger.debug(
                        "Custom dataset load failed for %s", file_path, exc_info=True
                    )
                else:
                    key = os.path.splitext(os.path.basename(file_path))[0]
                    datasets[key] = df

        else:
            corpus = self.dataset.get("corpus", None)
            config = self.dataset.get("config", None)
            dataset_path = os.path.join(base_path, f"{corpus}_{config}")
            os.makedirs(dataset_path, exist_ok=True)
            try:
                dataset = load_dataset(
                    corpus,
                    config,
                    cache_dir=dataset_path,
                    token=self.key_service.get_active_key(),
                )
            except Exception:
                logger.warning(
                    "Failed to download dataset %s with configuration %s",
                    corpus,
                    config,
                )
                logger.debug(
                    "Dataset download failure for %s/%s", corpus, config, exc_info=True
                )
                return None
            datasets[config] = dataset

        return datasets if datasets else None


# [DOWNLOADS]
###############################################################################
class TokenizersDownloadManager:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.key_service = HFAccessKeyService()
        self.tokenizers = configuration.get("TOKENIZERS", [])
        self.has_custom_tokenizer = configuration.get("include_custom_tokenizer", False)
        self.pipeline_tags = [
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

    def is_tokenizer_compatible(self, tokenizer: Any) -> bool:
        if tokenizer is None or isinstance(tokenizer, bool):
            return False

        if callable(getattr(tokenizer, "tokenize", None)):
            return True

        encode_method = getattr(tokenizer, "encode", None)
        decode_method = getattr(tokenizer, "decode", None)
        if callable(encode_method) and callable(decode_method):
            return True

        return False

    # -------------------------------------------------------------------------
    def get_tokenizer_identifiers(self, limit: int = 100, **kwargs) -> list[Any]:
        """
        Retrieve the most downloaded tokenizer identifiers from Hugging Face.

        Keyword arguments:
        limit -- maximum number of identifiers to request (default 100)

        Return value:
        List with the identifiers of the retrieved tokenizers ordered by
        popularity.
        """
        api = HfApi(token=self.key_service.get_active_key())
        try:
            models = api.list_models(
                search="tokenizer", sort="downloads", direction=-1, limit=limit
            )
        except Exception:
            logger.warning("Failed to retrieve tokenizer identifiers from HuggingFace")
            logger.debug("Tokenizer identifier fetch failed", exc_info=True)
            return []

        identifiers = [m.modelId for m in models]  # type: ignore

        return identifiers

    def _download_hf_tokenizer(self, tokenizer_id: str) -> Any | None:
        tokenizer_name = tokenizer_id.replace("/", "_")
        tokenizer_save_path = os.path.join(TOKENIZER_PATH, tokenizer_name)
        os.makedirs(tokenizer_save_path, exist_ok=True)
        logger.info(f"Downloading and saving tokenizer: {tokenizer_id}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_id,
            cache_dir=tokenizer_save_path,
            token=self.key_service.get_active_key(),
        )
        if not self.is_tokenizer_compatible(tokenizer):
            logger.warning(
                "Downloaded tokenizer %s is not compatible and will be skipped",
                tokenizer_id,
            )
            return None
        return tokenizer

    def _iter_custom_tokenizers(self, custom_tokenizer_path: str) -> list[tuple[str, Any]]:
        loaded_custom_tokenizers: list[tuple[str, Any]] = []
        json_files = [
            os.path.join(custom_tokenizer_path, fn)
            for fn in os.listdir(custom_tokenizer_path)
            if fn.lower().endswith(".json")
        ]
        if not json_files:
            return loaded_custom_tokenizers

        logger.info(f"Loading custom tokenizers from {custom_tokenizer_path}")
        for js in json_files:
            try:
                tokenizer = Tokenizer.from_file(js)
            except Exception:
                logger.warning("Failed to load custom tokenizer from %s", js)
                logger.debug("Custom tokenizer load failed for %s", js, exc_info=True)
                continue

            if not self.is_tokenizer_compatible(tokenizer):
                logger.warning(
                    "Custom tokenizer at %s is not compatible and will be skipped",
                    js,
                )
                continue

            tokenizer_name = os.path.basename(js).split(".")[0]
            loaded_custom_tokenizers.append((f"CUSTOM {tokenizer_name}", tokenizer))

        return loaded_custom_tokenizers

    # -------------------------------------------------------------------------
    def tokenizer_download(self, **kwargs) -> dict[str, Any]:
        """
        Download the tokenizers requested in the configuration and load any
        custom tokenizers stored locally.

        Keyword arguments:
        kwargs -- optional worker/progress references propagated from the GUI

        Return value:
        Dictionary mapping tokenizer identifiers to instantiated tokenizer
        objects ready for benchmarking.
        """
        tokenizers = {}
        for tokenizer_id in self.tokenizers:
            try:
                tokenizer = self._download_hf_tokenizer(tokenizer_id)
                if tokenizer is not None:
                    tokenizers[tokenizer_id] = tokenizer

            except Exception:
                logger.warning("Failed to download tokenizer %s", tokenizer_id)
                logger.debug(
                    "Tokenizer download error for %s", tokenizer_id, exc_info=True
                )

            finally:
                check_thread_status(kwargs.get("worker", None))

        # load custom tokenizer in target subfolder if .json files are found and
        # if the user has selected the option to include custom tokenizers
        custom_tokenizer_path = os.path.join(TOKENIZER_PATH, "custom")
        if os.path.exists(custom_tokenizer_path) and self.has_custom_tokenizer:
            check_thread_status(kwargs.get("worker", None))
            for tokenizer_name, tokenizer in self._iter_custom_tokenizers(
                custom_tokenizer_path
            ):
                tokenizers[tokenizer_name] = tokenizer

        return tokenizers

