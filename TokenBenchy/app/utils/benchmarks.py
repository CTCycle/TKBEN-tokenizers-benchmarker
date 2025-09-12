from __future__ import annotations

import os
import re
from typing import Any
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from tqdm import tqdm
from transformers.utils.logging import set_verbosity_error

from TokenBenchy.app.client.workers import check_thread_status, update_progress_callback
from TokenBenchy.app.constants import EVALUATION_PATH, TOKENIZER_PATH
from TokenBenchy.app.logger import logger
from TokenBenchy.app.utils.data.serializer import DataSerializer


# [TOKENIZERS EXPLORER]
###############################################################################
class BenchmarkTokenizers:
    def __init__(self, configuration: dict[str, Any]) -> None:
        set_verbosity_error()
        self.max_docs_number = configuration.get("num_documents", 0)
        self.reduce_data_size = configuration.get("reduce_output_size", False)
        self.include_custom_tokenizer = configuration.get(
            "include_custom_tokenizer", False
        )
        self.include_NSL = configuration.get("include_NSL", False)
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def process_tokens(self, text: str, tokenizer: Any) -> tuple[Any, Any]:
        ids = tokenizer.encode(text).ids
        decoded = tokenizer.decode(ids)
        toks = decoded.split()

        return decoded, toks

    # -------------------------------------------------------------------------
    def calculate_text_statistics(
        self, documents: pd.DataFrame, **kwargs
    ) -> pd.DataFrame | None:
        # interrupt the operation if no text dataset is available
        if documents.empty:
            logger.info("No text dataset available for statistics calculation")
            return

        dataset_name = documents["dataset_name"].iloc[0]
        logger.info(f"Loaded dataset {dataset_name} with {documents.shape[0]} records")

        max_documents = min(self.max_docs_number, len(documents))
        documents = documents[:max_documents] if max_documents > 0 else documents

        # 1/3 - Word count
        logger.info("Calculating word count for each document")
        documents["words_count"] = documents["text"].apply(lambda doc: len(doc.split()))

        check_thread_status(kwargs.get("worker", None))
        update_progress_callback(1, 3, kwargs.get("progress_callback", None))

        # 2/3 - Average word length
        logger.info("Calculating average word length for each document")
        documents["AVG_words_length"] = documents["text"].apply(
            lambda doc: np.mean([len(w) for w in doc.split()])
        )

        check_thread_status(kwargs.get("worker", None))
        update_progress_callback(2, 3, kwargs.get("progress_callback", None))

        # 3/3 - Standard deviation of word length
        logger.info("Calculating standard deviation of words length for each document")
        documents["STD_words_length"] = documents["text"].apply(
            lambda doc: np.std([len(w) for w in doc.split()])
        )

        check_thread_status(kwargs.get("worker", None))
        update_progress_callback(3, 3, kwargs.get("progress_callback", None))

        return documents

    # -------------------------------------------------------------------------
    def calculate_vocabulary_statistics(
        self, tokenizers: dict[str, Any], **kwargs
    ) -> tuple[list[Any], pd.DataFrame]:
        vocabulary_stats = []
        vocabularies = []
        # Iterate over each selected tokenizer from the combobox
        for i, (name, tokenizer) in enumerate(tokenizers.items()):
            vocab = tokenizer.get_vocab()
            vocab_words = list(vocab.keys())
            vocab_indices = list(vocab.values())
            # Identify subwords (words containing '##', typical for BERT-like tokenizers)
            subwords = [x for x in vocab_words if "##" in x]
            # Identify "true words" as elements that are not subwords
            true_words = [x for x in vocab_words if x not in subwords]
            # Decode the indices back to words
            decoded_words = tokenizer.decode(vocab_indices).split()
            # Identify tokens that are present in both the vocabulary and the decoded output
            shared = set(vocab_words).intersection(decoded_words)
            # Identify tokens that are in one but not both (symmetric difference)
            unshared = set(vocab_words).symmetric_difference(decoded_words)
            # Calculate percentage of subwords and true words in the vocabulary
            subwords_perc = len(subwords) / (len(true_words) + len(subwords)) * 100
            words_perc = len(true_words) / (len(true_words) + len(subwords)) * 100
            # Collect statistics for the current tokenizer
            vocabulary_stats.append(
                {
                    "tokenizer": name,
                    "number_tokens_from_vocabulary": len(vocab_words),
                    "number_tokens_from_decode": len(decoded_words),
                    "number_shared_tokens": len(shared),
                    "number_unshared_tokens": len(unshared),
                    "percentage_subwords": subwords_perc,
                    "percentage_true_words": words_perc,
                }
            )

            vocabulary = pd.DataFrame(
                {
                    "tokenizer": [name] * len(vocab_words),
                    "token_id": vocab_indices,
                    "vocabulary_tokens": vocab_words,
                    "decoded_tokens": [
                        tokenizer.decode([idx]) for idx in vocab_indices
                    ],
                }
            )

            # check for worker thread status and update progress callback
            check_thread_status(kwargs.get("worker", None))

            vocabularies.append(vocabulary)

        # save vocabulary statistics into database
        vocabulary_stats = pd.DataFrame(vocabulary_stats)

        return vocabularies, vocabulary_stats

    # -------------------------------------------------------------------------
    def run_tokenizer_benchmarks(
        self, dataset: pd.DataFrame, tokenizers: dict, **kwargs
    ) -> tuple[
        list[Any], pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame
    ]:
        """
        Run benchmarking for each tokenizer over a set of text documents.
        Optimized for speed and robustness, using vectorized operations where possible.
        Maintains original comments and logic.

        Args:
            tokenizers (dict): Dictionary of tokenizer name -> tokenizer instance.
            **kwargs: Optional arguments such as worker and progress_callback.
        Returns:
            pd.DataFrame: Concatenated benchmark results for all tokenizers.

        """

        # Calculate basic dataset statistics and extract vocabulary for each tokenizer
        vocabularies, vocabulary_stats = self.calculate_vocabulary_statistics(
            tokenizers, worker=kwargs.get("worker", None)
        )

        # Filter only a fraction of documents if requested by the user
        if 0 < self.max_docs_number <= len(dataset):
            dataset = dataset.iloc[: self.max_docs_number].reset_index(drop=True)

        # Prepare texts series only once for efficiency
        texts = dataset["text"].astype(str)
        num_docs = len(texts)
        all_tokenizers = []
        global_metrics_rows: list[dict[str, Any]] = []
        dataset_name = (
            str(dataset["dataset_name"].iloc[0])
            if "dataset_name" in dataset.columns and not dataset["dataset_name"].empty
            else ""
        )

        for i, (name, tokenizer) in enumerate(tokenizers.items()):
            k = name.replace("/", "_")
            logger.info(f"Decoding documents with {name}")

            # Initialize dataframe with tokenizer's name and dataset documents
            data = pd.DataFrame({"tokenizer": [name] * num_docs, "text": texts})

            # Calculate basic statistics using vectorized operations
            data["num_characters"] = texts.str.len()
            # Split words only once, keep as list for later calculations
            data["words_split"] = texts.str.split()
            data["words_count"] = data["words_split"].apply(len)
            data["AVG_words_length"] = data["words_split"].apply(
                lambda ws: np.mean([len(w) for w in ws]) if ws else 0
            )

            # time tokenization to compute speed and throughput
            t0 = time.perf_counter()
            if "CUSTOM" in name and self.include_custom_tokenizer:
                decoded_and_toks = data["text"].apply(
                    lambda text: pd.Series(self.process_tokens(text, tokenizer))
                )
                data["tokens"] = decoded_and_toks[0]
                data["tokens_split"] = decoded_and_toks[1]
            else:
                data["tokens_split"] = data["text"].apply(tokenizer.tokenize)
                data["tokens"] = data["tokens_split"].apply(
                    lambda toks: " ".join(toks)
                    if isinstance(toks, (list, tuple))
                    else ""
                )
            t1 = time.perf_counter()

            # Calculate number of tokens, token characters, average token length, ratios
            data["tokens_count"] = data["tokens_split"].apply(
                lambda toks: len(toks) if isinstance(toks, (list, tuple)) else 0
            )
            data["tokens_characters"] = data["tokens"].str.len()
            data["AVG_tokens_length"] = data["tokens_split"].apply(
                lambda toks: np.mean([len(tok) for tok in toks]) if toks else 0
            )
            # Use np.where for ratio calculations to avoid division by zero
            data["tokens_to_words_ratio"] = np.where(
                data["words_count"] > 0, data["tokens_count"] / data["words_count"], 0
            )
            data["bytes_per_token"] = np.where(
                data["tokens_count"] > 0,
                data["num_characters"] / data["tokens_count"],
                0,
            )

            # Compute per-tokenizer summary metrics (constants across rows for this tokenizer)
            elapsed = max(t1 - t0, 1e-9)
            total_tokens = int(data["tokens_count"].sum())
            total_chars = int(data["num_characters"].sum())
            tokenization_speed_tps = total_tokens / elapsed
            throughput_chars_per_sec = total_chars / elapsed

            # vocabulary size
            try:
                vocabulary_size = int(len(getattr(tokenizer, "get_vocab")()))
            except Exception:
                vocabulary_size = 0

            # model size: sum of files in cache dir if available
            model_size_mb = 0.0
            try:
                # try to infer path used during download (open tokenizers)
                base_dir = os.path.join(TOKENIZER_PATH, "open", k)
                if os.path.isdir(base_dir):
                    total_bytes = 0
                    for root, _dirs, files in os.walk(base_dir):
                        for fn in files:
                            fp = os.path.join(root, fn)
                            try:
                                total_bytes += os.path.getsize(fp)
                            except OSError:
                                pass
                    model_size_mb = total_bytes / (1024.0 * 1024.0)
            except Exception:
                model_size_mb = 0.0

            # average and median sequence length (tokens per document)
            seq_lengths = data["tokens_count"].to_numpy(dtype=float)
            avg_sequence_length = float(np.mean(seq_lengths)) if len(seq_lengths) else 0.0
            median_sequence_length = (
                float(np.median(seq_lengths)) if len(seq_lengths) else 0.0
            )

            # subword fertility: average tokens per word
            ww = data["words_count"].replace(0, np.nan)
            tw = data["tokens_count"]
            fertility_series = (tw / ww).replace([np.inf, -np.inf], np.nan).fillna(0)
            subword_fertility = float(fertility_series.mean())

            # OOV rate and word recovery rate
            # OOV: fraction of unique words not present as standalone tokens in vocab
            all_words = [w for lst in data["words_split"] for w in lst]
            unique_words = set(all_words)
            vocab_tokens = set()
            try:
                vocab_tokens = set(getattr(tokenizer, "get_vocab")().keys())
            except Exception:
                vocab_tokens = set()
            # strip common markers used by tokenizers when building character coverage
            normalized_vocab_tokens = {t.replace("##", "").lstrip("▁").lstrip("Ġ") for t in vocab_tokens}
            oov_words = {w for w in unique_words if w not in vocab_tokens}
            oov_rate = (len(oov_words) / len(unique_words) * 100.0) if unique_words else 0.0

            # word recovery: proportion of words where decode(encode(word)) == word
            # evaluate on a sample to keep it fast
            recovery_count = 0
            sample_words = list(unique_words)
            max_eval = min(5000, len(sample_words))
            sample_words = sample_words[:max_eval]
            for w in sample_words:
                try:
                    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
                        enc = tokenizer.encode(w)
                        # support both HF tokenizers and transformers
                        if hasattr(enc, "ids"):
                            dec = tokenizer.decode(enc.ids)
                        else:
                            dec = tokenizer.decode(enc)
                        if dec == w:
                            recovery_count += 1
                except Exception:
                    continue
            word_recovery_rate = (recovery_count / max(1, len(sample_words))) * 100.0

            # character coverage: unique chars in dataset covered by vocab tokens
            dataset_chars = set("".join(texts.tolist()))
            vocab_chars = set()
            for tkn in normalized_vocab_tokens:
                for ch in tkn:
                    vocab_chars.add(ch)
            intersection = dataset_chars.intersection(vocab_chars)
            character_coverage = (
                (len(intersection) / len(dataset_chars) * 100.0) if dataset_chars else 0.0
            )

            # attach per-tokenizer metrics to each row for this tokenizer
            # accumulate global metrics row (one per tokenizer)
            global_metrics_rows.append(
                {
                    "tokenizer": name,
                    "dataset_name": dataset_name,
                    "tokenization_speed_tps": float(tokenization_speed_tps),
                    "throughput_chars_per_sec": float(throughput_chars_per_sec),
                    "model_size_mb": float(model_size_mb),
                    "vocabulary_size": int(vocabulary_size),
                    "avg_sequence_length": float(avg_sequence_length),
                    "median_sequence_length": float(median_sequence_length),
                    "subword_fertility": float(subword_fertility),
                    "oov_rate": float(oov_rate),
                    "word_recovery_rate": float(word_recovery_rate),
                    "character_coverage": float(character_coverage),
                }
            )

            # Drop intermediate columns if reduce_data_size is set
            drop_cols = ["tokens", "tokens_split", "words_split"]
            data.drop(columns=drop_cols, inplace=True)
            all_tokenizers.append(data)

            # Progress update and thread safety
            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(tokenizers), kwargs.get("progress_callback", None)
            )

        # Concatenate all tokenizer benchmark results (local per-text stats)
        benchmark_results = pd.concat(all_tokenizers, ignore_index=True)

        # Calculate NSL if required
        data_NSL = None
        if self.include_NSL and self.include_custom_tokenizer:
            self.calculate_normalized_sequence_length(benchmark_results)

        global_metrics = pd.DataFrame(global_metrics_rows)

        return vocabularies, vocabulary_stats, benchmark_results, data_NSL, global_metrics

    # -------------------------------------------------------------------------
    def calculate_normalized_sequence_length(
        self, benchmark_results: pd.DataFrame
    ) -> None | pd.DataFrame:
        data_custom = benchmark_results[
            benchmark_results["tokenizer"].str.contains(
                "custom tokenizer", case=False, na=False
            )
        ]

        data = []
        names = list(benchmark_results["tokenizer"].unique())
        if data_custom.empty:
            logger.warning(
                "NSL value cannot be calculated without a custom tokenizer as reference"
            )
            return None
        else:
            for tok in tqdm(names):
                logger.info(
                    f"NSL value is calculated for {tok} versus custom tokenizers"
                )
                data_chunk = benchmark_results[benchmark_results["tokenizer"] == tok]
                data_chunk["NSL"] = [
                    x / y if y != 0 else 0
                    for x, y in zip(
                        data_custom["tokens_count"].to_list(),
                        data_chunk["tokens_count"].to_list(),
                    )
                ]
                data.append(data_chunk)

            data_NSL = pd.concat(data, ignore_index=True)

        return data_NSL


# [TOKENIZERS EXPLORER]
###############################################################################
class VisualizeBenchmarkResults:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.img_resolution = 400
        self.observed_features = [
            "tokens_to_words_ratio",
            "AVG_tokens_length",
            "bytes_per_token",
        ]
        self.configuration = configuration
        self.serializer = DataSerializer()

    # -------------------------------------------------------------------------
    def save_image(self, fig: Figure, name: str) -> None:
        name = re.sub(r"[^0-9A-Za-z_]", "_", name)
        out_path = os.path.join(EVALUATION_PATH, name)
        fig.savefig(out_path, bbox_inches="tight", dpi=self.img_resolution)

    # -------------------------------------------------------------------------
    def plot_vocabulary_size(self, data: pd.DataFrame) -> Figure:
        df = data.dropna(subset=["vocabulary_tokens"])
        df["vocabulary_tokens"] = df["vocabulary_tokens"].astype(str)
        df = df[df["vocabulary_tokens"].str.len() > 0]

        counts = (
            df.groupby("tokenizer", sort=False)["vocabulary_tokens"]
            .nunique()
            .sort_values(ascending=True)
        )

        n_tok = len(counts)
        width_in = 14
        height_in = max(4, 0.7 * n_tok + 2)

        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=self.img_resolution)

        # Use numeric positions + set tick labels; convert widths to NumPy to avoid ExtensionArray typing issues.
        y_pos = np.arange(n_tok, dtype=float)
        widths = counts.to_numpy(dtype=float)

        ax.barh(y_pos, widths, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(counts.index))

        ax.set_title(
            "Vocabulary size by tokenizer",
            fontsize=16,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Number of tokens", fontsize=13, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        fig.tight_layout()
        self.save_image(fig, "vocabulary_size.jpeg")
        plt.close(fig)
        return fig

    # -------------------------------------------------------------------------
    def plot_subwords_vs_words(self, data: pd.DataFrame) -> Figure | None:
        df = data.loc[:, ["tokenizer", "vocabulary_tokens"]].copy()
        df = df.dropna(subset=["tokenizer", "vocabulary_tokens"])
        if df.empty:
            logger.info("plot_subwords_vs_words: no data to plot")
            return None

        # Normalize types and trim whitespace, sanitize all strings to avoid ambiguity
        df["tokenizer"] = df["tokenizer"].astype(str)
        df["vocabulary_tokens"] = df["vocabulary_tokens"].astype(str).str.strip()
        df = df[df["vocabulary_tokens"].str.len() > 0]
        if df.empty:
            return None

        # Special tokens to exclude from both categories
        special_pat = r"^(?:\[.*\]|<.*>|{.*}|</?s>|</?pad>|UNK|PAD)$"
        is_special = df["vocabulary_tokens"].str.match(special_pat, case=False)
        df = df[~is_special]
        if df.empty:
            logger.info("plot_subwords_vs_words: only special tokens present")
            return None

        # identify BERT subwords starting with '##'
        bert_sub = df["vocabulary_tokens"].str.startswith("##")

        # identify SentencePiece rule subwords, where leading '▁' denotes word start
        # tokens with '▁' in other positions are subwords
        sp_has = df["vocabulary_tokens"].str.contains("▁", regex=False)
        sp_word_start = df["vocabulary_tokens"].str.startswith("▁")
        sp_sub = sp_has & (~sp_word_start)

        # identify GPT-2/BBPE rule subwords, where leading 'Ġ' denotes word start
        bbpe_has = df["vocabulary_tokens"].str.contains("Ġ", regex=False)
        bbpe_word_start = df["vocabulary_tokens"].str.startswith("Ġ")
        bbpe_sub = bbpe_has & (~bbpe_word_start)

        is_subword = bert_sub | sp_sub | bbpe_sub
        is_word = ~is_subword

        # Aggregate counts per tokenizer
        df["is_subword"] = is_subword.astype(int)
        df["is_word"] = is_word.astype(int)

        grouped = (
            df.groupby("tokenizer", sort=False)
            .agg(subwords_count=("is_subword", "sum"), words_count=("is_word", "sum"))
            .reset_index()
        )

        # Sort by subwords count
        grouped = grouped.sort_values("subwords_count", ascending=False).reset_index(
            drop=True
        )

        if grouped.empty:
            logger.info("plot_subwords_vs_words: nothing to plot after aggregation")
            return None

        tokenizers = grouped["tokenizer"].tolist()
        n = len(tokenizers)
        x = np.arange(n, dtype=float)
        width = 0.4

        subwords = grouped["subwords_count"].to_numpy(dtype=float)
        words = grouped["words_count"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 7), dpi=self.img_resolution)
        ax.bar(
            x - width / 2,
            subwords,
            width=width,
            edgecolor="black",
            color="#105D8D",
            label="Subwords (vocabulary)",
        )
        ax.bar(
            x + width / 2,
            words,
            width=width,
            edgecolor="black",
            color="#107F40",
            label="Words (vocabulary)",
        )

        ax.set_title(
            "Subwords vs Words by tokenizer (vocabulary)",
            fontsize=16,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Tokenizer", fontsize=13, fontweight="bold")
        ax.set_ylabel("Count", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tokenizers, rotation=35, ha="right")

        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(title="", fontsize=11)

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            try:
                lbl.set_fontweight("bold")
            except Exception:
                pass

        fig.tight_layout()
        self.save_image(fig, "subwords_vs_words.jpeg")
        plt.close(fig)
        return fig

    # -------------------------------------------------------------------------
    def plot_tokens_length_distribution(self, data: pd.DataFrame) -> list[Any]:
        figures = []
        # Preserve tokenizer order as encountered
        tokenizers = data["tokenizer"].astype(str).dropna().unique().tolist()
        for tk in tokenizers:
            df_tk = data[data["tokenizer"] == tk]
            # Prepare series and compute lengths (exclude NaNs and empty strings)
            vocab_series = df_tk["vocabulary_tokens"].dropna().astype(str)
            decoded_series = df_tk["decoded_tokens"].dropna().astype(str)

            vocab_lengths = (
                vocab_series[vocab_series.str.len() > 0].str.len().to_numpy(dtype=int)
            )
            decoded_lengths = (
                decoded_series[decoded_series.str.len() > 0]
                .str.len()
                .to_numpy(dtype=int)
            )

            # Determine common integer-aligned bins across both panels for comparability
            max_len = int(
                max(
                    (vocab_lengths.max() if vocab_lengths.size else 0),
                    (decoded_lengths.max() if decoded_lengths.size else 0),
                )
            )

            # Create figure
            fig, axs = plt.subplots(
                2, 1, figsize=(16, 10), dpi=self.img_resolution, sharex=True
            )

            if max_len == 0:
                titles = (
                    f"{tk} • Vocabulary token length distribution",
                    f"{tk} • Decoded token length distribution",
                )
                for ax, title in zip(axs, titles):
                    ax.text(
                        0.5, 0.5, "No token data", ha="center", va="center", fontsize=14
                    )
                    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
                    ax.set_xlabel(
                        "Token length (chars)", fontsize=13, fontweight="bold"
                    )
                    ax.set_ylabel("Frequency", fontsize=13, fontweight="bold")
                    ax.grid(axis="both", linestyle="--", alpha=0.3)

                fig.tight_layout()
                # Save and collect
                self.save_image(fig, f"{tk}_token_length_hist.jpeg")
                figures.append(fig)
                plt.close(fig)
                continue

            # Integer-centered bins: [0.5, 1.5, ..., max_len + 0.5]
            bin_edges = np.arange(0.5, max_len + 1.5, 1.0, dtype=float)

            # Panel 1 — vocabulary token lengths
            axs[0].hist(vocab_lengths, bins=bin_edges, edgecolor="black")
            axs[0].set_title(
                f"{tk} • Vocabulary token length distribution",
                fontsize=16,
                fontweight="bold",
                pad=10,
            )
            axs[0].set_ylabel("Frequency", fontsize=13, fontweight="bold")
            axs[0].grid(axis="both", linestyle="--", alpha=0.3)

            # Panel 2 — decoded token lengths
            axs[1].hist(decoded_lengths, bins=bin_edges, edgecolor="black")
            axs[1].set_title(
                f"{tk} • Decoded token length distribution",
                fontsize=16,
                fontweight="bold",
                pad=10,
            )
            axs[1].set_xlabel("Token length (chars)", fontsize=13, fontweight="bold")
            axs[1].set_ylabel("Frequency", fontsize=13, fontweight="bold")
            axs[1].grid(axis="both", linestyle="--", alpha=0.3)

            # Ticks & layout
            axs[1].set_xticks(np.arange(1, max_len + 1, 1))
            for ax in axs:
                ax.tick_params(axis="x", labelsize=11)
                ax.tick_params(axis="y", labelsize=11)
                # Embolden tick labels for readability
                for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                    try:
                        lbl.set_fontweight("bold")
                    except Exception:
                        pass

            fig.tight_layout()

            # Save and collect
            self.save_image(fig, f"{tk}_token_length_hist.jpeg")
            figures.append(fig)
            plt.close(fig)

        return figures
