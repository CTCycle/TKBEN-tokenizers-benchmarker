from __future__ import annotations

import os
from typing import Any

import pandas as pd
import transformers
from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer

from TokenBenchy.app.client.workers import check_thread_status
from TokenBenchy.app.constants import DATASETS_PATH, TOKENIZER_PATH
from TokenBenchy.app.logger import logger


# [DOWNLOADS]
###############################################################################
class DatasetManager:
    def __init__(
        self, configuration: dict[str, Any], hf_access_token: str | None
    ) -> None:
        self.dataset = configuration.get("DATASET", {})
        self.dataset_corpus = self.dataset.get("corpus", "wikitext")
        self.dataset_config = self.dataset.get("config", "wikitext-103-v1")
        self.has_custom_dataset = configuration.get("use_custom_dataset", False)
        self.configuration = configuration
        self.hf_access_token = hf_access_token

    # -------------------------------------------------------------------------
    def get_dataset_name(self) -> str | Any:
        if self.dataset_config:
            return f"{self.dataset_corpus}/{self.dataset_config}"
        return self.dataset_corpus

    # -------------------------------------------------------------------------
    def dataset_download(self) -> None | dict[str, Any]:
        datasets = {}
        subfolder = "custom" if self.has_custom_dataset else "open"
        base_path = os.path.join(DATASETS_PATH, subfolder)
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
                df = pd.read_csv(file_path)
                key = os.path.splitext(os.path.basename(file_path))[0]
                datasets[key] = df

        else:
            corpus = self.dataset.get("corpus", None)
            config = self.dataset.get("config", None)
            dataset_path = os.path.join(base_path, f"{corpus}_{config}")
            os.makedirs(dataset_path, exist_ok=True)
            dataset = load_dataset(corpus, config, cache_dir=dataset_path)
            datasets[config] = dataset

        return datasets


# [DOWNLOADS]
###############################################################################
class TokenizersDownloadManager:
    def __init__(
        self, configuration: dict[str, Any], hf_access_token: str | None
    ) -> None:
        self.hf_access_token = hf_access_token
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

    # -------------------------------------------------------------------------
    def get_tokenizer_identifiers(self, limit: int = 100, **kwargs) -> list[Any]:
        api = HfApi(token=self.hf_access_token) if "downloads" else HfApi()
        # query HuggingFace Hub to search for tokenizer tag in metadata, sort by downloads
        models = api.list_models(
            search="tokenizer", sort="downloads", direction=-1, limit=limit
        )
        # extract and return just the model IDs
        identifiers = [m.model_index for m in models]

        return identifiers

    # -------------------------------------------------------------------------
    def tokenizer_download(self, **kwargs) -> dict[str, Any]:
        tokenizers = {}
        for tokenizer_id in self.tokenizers:
            try:
                tokenizer_name = tokenizer_id.replace("/", "_")
                tokenizer_save_path = os.path.join(
                    TOKENIZER_PATH, "open", tokenizer_name
                )
                os.mkdir(tokenizer_save_path) if not os.path.exists(
                    tokenizer_save_path
                ) else None
                logger.info(f"Downloading and saving tokenizer: {tokenizer_id}")
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer_id,
                    cache_dir=tokenizer_save_path,
                    token=self.hf_access_token,
                )
                tokenizers[tokenizer_id] = tokenizer

            except Exception as e:
                logger.warning(f"Failed to download tokenizer {tokenizer_id}")
                logger.debug(
                    f"Failed to download tokenizer {tokenizer_id}: {e}", exc_info=True
                )

            finally:
                # check for worker thread status
                check_thread_status(kwargs.get("worker", None))

        # load custom tokenizer in target subfolder if .json files are found and
        # if the user has selected the option to include custom tokenizers
        custom_tokenizer_path = os.path.join(TOKENIZER_PATH, "custom")
        if os.path.exists(custom_tokenizer_path) and self.has_custom_tokenizer:
            # check for worker thread status
            check_thread_status(kwargs.get("worker", None))
            json_files = [
                os.path.join(custom_tokenizer_path, fn)
                for fn in os.listdir(custom_tokenizer_path)
                if fn.lower().endswith(".json")
            ]

            if len(json_files) > 0 and self.has_custom_tokenizer:
                logger.info(f"Loading custom tokenizers from {custom_tokenizer_path}")
                for js in json_files:
                    tokenizer = Tokenizer.from_file(js)
                    tokenizer_name = os.path.basename(js).split(".")[0]
                    tokenizers[f"CUSTOM {tokenizer_name}"] = tokenizer

        return tokenizers
