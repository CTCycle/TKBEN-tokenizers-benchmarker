from __future__ import annotations

import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

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
class BenchmarkTools:
    # -------------------------------------------------------------------------
    def process_tokens(self, text: str, tokenizer: Any) -> tuple[str, list[str]]:
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return "", []

        tokenizer_name = getattr(tokenizer, "name_or_path", type(tokenizer).__name__)

        try:
            encoded = tokenizer.encode(text)
        except Exception:
            logger.debug(
                "Tokenizer %s raised an exception while encoding text",
                tokenizer_name,
                exc_info=True,
            )
            return "", []

        tokens_from_encoding: list[str] = []
        if hasattr(encoded, "tokens"):
            tokens_attr = getattr(encoded, "tokens")
            if callable(tokens_attr):
                try:
                    raw_tokens = tokens_attr()
                except Exception:
                    logger.debug(
                        "Tokenizer %s failed to extract tokens from encoding",
                        tokenizer_name,
                        exc_info=True,
                    )
                else:
                    tokens_from_encoding = self.normalize_token_output(raw_tokens)
            elif isinstance(tokens_attr, (list, tuple)):
                tokens_from_encoding = [str(tok) for tok in tokens_attr]
            elif isinstance(tokens_attr, Iterable) and not isinstance(
                tokens_attr, (str, bytes)
            ):
                tokens_from_encoding = [str(tok) for tok in tokens_attr]

        token_ids = self.extract_token_ids(encoded)
        if token_ids:
            decoded = self.safe_decode(tokenizer, token_ids)
        else:
            decoded = " ".join(tokens_from_encoding)

        if not tokens_from_encoding:
            tokens_from_encoding = self.convert_ids_to_tokens(
                tokenizer, token_ids, decoded
            )

        return decoded, tokens_from_encoding

    # -------------------------------------------------------------------------
    def extract_token_ids(self, encoded: Any) -> list[int]:
        ids_source: Any | None = None
        if hasattr(encoded, "ids"):
            ids_source = getattr(encoded, "ids")
        elif isinstance(encoded, np.ndarray):
            ids_source = encoded.tolist()
        elif isinstance(encoded, (list, tuple)):
            ids_source = encoded

        if ids_source is None:
            return []

        try:
            return [int(i) for i in ids_source]
        except Exception:
            logger.debug("Failed to coerce token ids from encoding", exc_info=True)
            return []

    # -------------------------------------------------------------------------
    def safe_decode(self, tokenizer: Any, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        try:
            decoded = tokenizer.decode(token_ids)
        except Exception:
            logger.debug(
                "Tokenizer %s raised an exception while decoding ids",
                getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
                exc_info=True,
            )
            return ""

        if isinstance(decoded, (list, tuple)):
            return " ".join(str(tok) for tok in decoded)
        return str(decoded)

    # -------------------------------------------------------------------------
    def convert_ids_to_tokens(
        self, tokenizer: Any, token_ids: list[int], fallback_text: str
    ) -> list[str]:
        try:
            converter = getattr(tokenizer, "convert_ids_to_tokens", None)
            if callable(converter):
                tokens = converter(token_ids)
                if isinstance(tokens, np.ndarray):
                    tokens = tokens.tolist()
                if isinstance(tokens, (list, tuple)):
                    return [str(tok) for tok in tokens]
            id_to_token = getattr(tokenizer, "id_to_token", None)
            if callable(id_to_token):
                return [str(id_to_token(idx)) for idx in token_ids]
        except Exception:
            logger.debug(
                "Tokenizer %s failed to convert ids to tokens",
                getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
                exc_info=True,
            )

        if fallback_text:
            return fallback_text.split()
        return []

    # -------------------------------------------------------------------------
    def normalize_token_output(self, tokens: Any) -> list[str]:
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        if hasattr(tokens, "tokens") and callable(getattr(tokens, "tokens")):
            try:
                tokens = tokens.tokens()
            except Exception:
                logger.debug("Failed to normalize token output", exc_info=True)
                return []

        if isinstance(tokens, (list, tuple)):
            return [str(tok) for tok in tokens]

        if isinstance(tokens, str):
            return tokens.split()

        if isinstance(tokens, dict):
            for key in ("tokens", "input_tokens"):
                value = tokens.get(key)
                if isinstance(value, (list, tuple)):
                    return [str(tok) for tok in value]

        if isinstance(tokens, Iterable):
            return [str(tok) for tok in tokens]

        return []

    # -------------------------------------------------------------------------
    def is_tokenizer_compatible(self, tokenizer: Any) -> bool:
        if tokenizer is None or isinstance(tokenizer, bool):
            return False

        if callable(getattr(tokenizer, "tokenize", None)):
            return True

        encode_method = getattr(tokenizer, "encode", None)
        decode_method = getattr(tokenizer, "decode", None)
        if callable(encode_method) and callable(decode_method):
            return True

        call_method = getattr(tokenizer, "__call__", None)
        return callable(call_method)


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
        self.tools = BenchmarkTools()

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
        vocabulary_stats: list[dict[str, Any]] = []
        vocabularies: list[pd.DataFrame] = []
        for name, tokenizer in tokenizers.items():
            if not self.tools.is_tokenizer_compatible(tokenizer):
                logger.warning(
                    'Skipping tokenizer %s because it does not expose required methods (type=%s)',
                    name,
                    type(tokenizer).__name__,
                )
                continue

            tokenizer_label = getattr(tokenizer, 'name_or_path', name)
            try:
                vocab_func = getattr(tokenizer, 'get_vocab', None)
                if not callable(vocab_func):
                    logger.warning(
                        'Tokenizer %s does not expose get_vocab; skipping vocabulary statistics',
                        tokenizer_label,
                    )
                    continue

                raw_vocab = vocab_func()
                if not isinstance(raw_vocab, dict):
                    logger.warning(
                        'Tokenizer %s returned unexpected vocabulary of type %s',
                        tokenizer_label,
                        type(raw_vocab).__name__,
                    )
                    continue

                vocab_words = [str(word) for word in raw_vocab.keys()]
                try:
                    vocab_indices = [int(idx) for idx in raw_vocab.values()]
                except Exception:
                    logger.debug(
                        'Tokenizer %s produced non-numeric vocab indices',
                        tokenizer_label,
                        exc_info=True,
                    )
                    continue

                subwords = [word for word in vocab_words if '##' in word]
                true_words = [word for word in vocab_words if word not in subwords]
                decoded_words = self.tools.safe_decode(tokenizer, vocab_indices).split()
                shared = set(vocab_words).intersection(decoded_words)
                unshared = set(vocab_words).symmetric_difference(decoded_words)
                total_tokens = len(true_words) + len(subwords)
                subwords_perc = (len(subwords) / total_tokens * 100.0) if total_tokens else 0.0
                words_perc = (len(true_words) / total_tokens * 100.0) if total_tokens else 0.0

                vocabulary_stats.append(
                    {
                        'tokenizer': name,
                        'number_tokens_from_vocabulary': len(vocab_words),
                        'number_tokens_from_decode': len(decoded_words),
                        'number_shared_tokens': len(shared),
                        'number_unshared_tokens': len(unshared),
                        'percentage_subwords': subwords_perc,
                        'percentage_true_words': words_perc,
                    }
                )

                decoded_per_id = [
                    self.tools.safe_decode(tokenizer, [idx]) for idx in vocab_indices
                ]
                vocabulary = pd.DataFrame(
                    {
                        'tokenizer': [name] * len(vocab_words),
                        'token_id': vocab_indices,
                        'vocabulary_tokens': vocab_words,
                        'decoded_tokens': decoded_per_id,
                    }
                )

                check_thread_status(kwargs.get('worker', None))
                vocabularies.append(vocabulary)
            except Exception:
                logger.warning(f'Could not process tokenizer {name}')
                logger.debug(
                    'Failed to collect vocabulary statistics for tokenizer %s',
                    name,
                    exc_info=True,
                )
                continue

        vocabulary_stats_df = pd.DataFrame(vocabulary_stats)

        return vocabularies, vocabulary_stats_df

    # -------------------------------------------------------------------------
    def run_tokenizer_benchmarks(
        self, dataset: pd.DataFrame, tokenizers: dict, **kwargs
    ) -> tuple[
        list[Any], pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame
    ]:
        valid_tokenizers: dict[str, Any] = {}
        if isinstance(tokenizers, dict):
            for name, tokenizer in tokenizers.items():
                if self.tools.is_tokenizer_compatible(tokenizer):
                    valid_tokenizers[name] = tokenizer
                else:
                    logger.warning(
                        'Skipping tokenizer %s because it does not expose required methods (type=%s)',
                        name,
                        type(tokenizer).__name__,
                    )
        else:
            logger.warning('Tokenizers input is not a dictionary; skipping benchmarks')

        if not valid_tokenizers:
            logger.warning('No valid tokenizers available for benchmarking')
            empty_df = pd.DataFrame()
            return [], pd.DataFrame(), empty_df, None, pd.DataFrame()

        if dataset is None or dataset.empty:
            logger.warning('No dataset available for benchmarking')
            empty_df = pd.DataFrame()
            return [], pd.DataFrame(), empty_df, None, pd.DataFrame()

        if 'text' not in dataset.columns:
            logger.warning("Dataset does not contain required 'text' column; skipping benchmarks")
            empty_df = pd.DataFrame()
            return [], pd.DataFrame(), empty_df, None, pd.DataFrame()

        vocabularies, vocabulary_stats = self.calculate_vocabulary_statistics(
            valid_tokenizers, worker=kwargs.get('worker', None)
        )

        dataset_to_use = dataset
        if 0 < self.max_docs_number <= len(dataset_to_use):
            dataset_to_use = dataset_to_use.iloc[: self.max_docs_number].reset_index(
                drop=True
            )

        texts = dataset_to_use['text'].astype(str)
        num_docs = len(texts)
        all_tokenizers: list[pd.DataFrame] = []
        global_metrics_rows: list[dict[str, Any]] = []
        dataset_name = (
            str(dataset_to_use['dataset_name'].iloc[0])
            if 'dataset_name' in dataset_to_use.columns
            and not dataset_to_use['dataset_name'].empty
            else ''
        )

        progress_total = len(valid_tokenizers)
        for i, (name, tokenizer) in enumerate(valid_tokenizers.items()):
            text_values = texts.tolist()
            logger.info(f'Decoding documents with {name}')

            data = pd.DataFrame(
                {
                    'tokenizer': [name] * num_docs,
                    'text': text_values,
                }
            )

            data['num_characters'] = texts.str.len()
            data['words_split'] = texts.str.split()
            data['words_count'] = data['words_split'].apply(len)
            data['AVG_words_length'] = data['words_split'].apply(
                lambda ws: np.mean([len(w) for w in ws]) if ws else 0
            )

            t0 = time.perf_counter()
            try:
                if 'CUSTOM' in name and self.include_custom_tokenizer:
                    decoded_tokens: list[str] = []
                    split_tokens: list[list[str]] = []
                    for text_value in text_values:
                        decoded, tokens_list = self.tools.process_tokens(
                            text_value, tokenizer
                        )
                        decoded_tokens.append(decoded)
                        split_tokens.append(tokens_list)
                else:
                    tokenize_method = getattr(tokenizer, 'tokenize', None)
                    uses_tokenize = callable(tokenize_method)
                    decoded_tokens = []
                    split_tokens = []
                    for text_value in text_values:
                        tokens_list: list[str] = []
                        if uses_tokenize:
                            try:
                                raw_tokens = tokenize_method(text_value)  # type: ignore[operator]
                                tokens_list = self.tools.normalize_token_output(raw_tokens)
                            except Exception:
                                logger.debug(
                                    'Tokenizer %s raised an exception while tokenizing text',
                                    name,
                                    exc_info=True,
                                )
                                tokens_list = []
                        if not tokens_list:
                            decoded, tokens_list = self.tools.process_tokens(
                                text_value, tokenizer
                            )
                        else:
                            decoded = ' '.join(tokens_list)
                        decoded_tokens.append(decoded)
                        split_tokens.append(tokens_list)

                data['tokens'] = decoded_tokens
                data['tokens_split'] = split_tokens
            except Exception:
                logger.warning('Failed to tokenize documents with %s', name)
                logger.debug(
                    'Tokenizer %s raised an exception during batch tokenization',
                    name,
                    exc_info=True,
                )
                check_thread_status(kwargs.get('worker', None))
                update_progress_callback(
                    i + 1, progress_total, kwargs.get('progress_callback', None)
                )
                continue

            t1 = time.perf_counter()

            data['tokens_count'] = [
                len(toks) if isinstance(toks, (list, tuple)) else 0
                for toks in data['tokens_split']
            ]
            data['tokens_characters'] = data['tokens'].str.len()
            data['AVG_tokens_length'] = data['tokens_split'].apply(
                lambda toks: np.mean([len(tok) for tok in toks]) if toks else 0
            )
            data['tokens_to_words_ratio'] = np.where(
                data['words_count'] > 0, data['tokens_count'] / data['words_count'], 0
            )
            data['bytes_per_token'] = np.where(
                data['tokens_count'] > 0,
                data['num_characters'] / data['tokens_count'],
                0,
            )

            elapsed = max(t1 - t0, 1e-9)
            total_tokens = int(data['tokens_count'].sum())
            total_chars = int(data['num_characters'].sum())
            tokenization_speed_tps = total_tokens / elapsed
            throughput_chars_per_sec = total_chars / elapsed

            vocab_method = getattr(tokenizer, 'get_vocab', None)
            vocab_result: Mapping[Any, Any] | Sequence[Any] | None = None
            if callable(vocab_method):
                try:
                    candidate = vocab_method()
                except Exception:
                    logger.debug(
                        'Tokenizer %s failed to expose its vocabulary during metrics computation',
                        name,
                        exc_info=True,
                    )
                else:
                    if isinstance(candidate, Mapping):
                        vocab_result = candidate
                    elif isinstance(candidate, Sequence) and not isinstance(
                        candidate, (str, bytes)
                    ):
                        vocab_result = candidate

            if isinstance(vocab_result, Mapping):
                vocabulary_size = int(len(vocab_result))
            elif isinstance(vocab_result, Sequence):
                vocabulary_size = int(len(vocab_result))
            else:
                vocabulary_size = 0

            model_size_mb = 0.0
            try:
                base_dir = os.path.join(TOKENIZER_PATH, 'open', name.replace('/', '_'))
                if os.path.isdir(base_dir):
                    total_bytes = 0
                    for root, _dirs, files in os.walk(base_dir):
                        for fn in files:
                            fp = os.path.join(root, fn)
                            try:
                                total_bytes += os.path.getsize(fp)
                            except OSError:
                                logger.debug('Failed to read file size for %s', fp, exc_info=True)
                    model_size_mb = total_bytes / (1024.0 * 1024.0)
            except Exception:
                logger.debug(
                    'Tokenizer %s raised an exception while calculating model size',
                    name,
                    exc_info=True,
                )
                model_size_mb = 0.0

            seq_lengths = data['tokens_count'].to_numpy(dtype=float)
            avg_sequence_length = float(np.mean(seq_lengths)) if len(seq_lengths) else 0.0
            median_sequence_length = (
                float(np.median(seq_lengths)) if len(seq_lengths) else 0.0
            )

            words_per_doc = data['words_count'].replace(0, np.nan)
            tokens_per_doc = data['tokens_count']
            fertility_series = (tokens_per_doc / words_per_doc).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0)
            subword_fertility = float(fertility_series.mean())

            all_words = [w for lst in data['words_split'] for w in lst]
            unique_words = set(all_words)
            vocab_tokens: set[str] = set()
            if isinstance(vocab_result, Mapping):
                vocab_tokens = {str(tok) for tok in vocab_result.keys()}
            elif isinstance(vocab_result, Sequence):
                vocab_tokens = {str(tok) for tok in vocab_result}
            normalized_vocab_tokens = {
                str(t).replace('##', '').lstrip('?').lstrip('G') for t in vocab_tokens
            }
            oov_words = {w for w in unique_words if w not in vocab_tokens}
            oov_rate = (len(oov_words) / len(unique_words) * 100.0) if unique_words else 0.0

            recovery_count = 0
            sample_words = list(unique_words)
            max_eval = min(5000, len(sample_words))
            sample_words = sample_words[:max_eval]
            for w in sample_words:
                try:
                    if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'decode'):
                        enc = tokenizer.encode(w)
                        if hasattr(enc, 'ids'):
                            dec = tokenizer.decode(enc.ids)
                        else:
                            dec = tokenizer.decode(enc)
                        if dec == w:
                            recovery_count += 1
                except Exception:
                    continue
            word_recovery_rate = (recovery_count / max(1, len(sample_words))) * 100.0

            dataset_chars = set(''.join(text_values))
            vocab_chars = set()
            for token_item in normalized_vocab_tokens:
                for ch in token_item:
                    vocab_chars.add(ch)
            intersection = dataset_chars.intersection(vocab_chars)
            character_coverage = (
                (len(intersection) / len(dataset_chars) * 100.0) if dataset_chars else 0.0
            )

            global_metrics_rows.append(
                {
                    'tokenizer': name,
                    'dataset_name': dataset_name,
                    'tokenization_speed_tps': float(tokenization_speed_tps),
                    'throughput_chars_per_sec': float(throughput_chars_per_sec),
                    'model_size_mb': float(model_size_mb),
                    'vocabulary_size': int(vocabulary_size),
                    'avg_sequence_length': float(avg_sequence_length),
                    'median_sequence_length': float(median_sequence_length),
                    'subword_fertility': float(subword_fertility),
                    'oov_rate': float(oov_rate),
                    'word_recovery_rate': float(word_recovery_rate),
                    'character_coverage': float(character_coverage),
                }
            )

            if self.reduce_data_size:
                data.drop(columns=['tokens', 'tokens_split', 'words_split'], inplace=True)
            all_tokenizers.append(data)

            check_thread_status(kwargs.get('worker', None))
            update_progress_callback(
                i + 1, progress_total, kwargs.get('progress_callback', None)
            )

        benchmark_results = (
            pd.concat(all_tokenizers, ignore_index=True) if all_tokenizers else pd.DataFrame()
        )

        data_NSL = None
        if (
            self.include_NSL
            and self.include_custom_tokenizer
            and not benchmark_results.empty
        ):
            data_NSL = self.calculate_normalized_sequence_length(benchmark_results)

        global_metrics = pd.DataFrame(global_metrics_rows)

        return vocabularies, vocabulary_stats, benchmark_results, data_NSL, global_metrics

    # -------------------------------------------------------------------------
    def calculate_normalized_sequence_length(
        self, benchmark_results: pd.DataFrame
    ) -> None | pd.DataFrame:
        if benchmark_results is None or benchmark_results.empty:
            logger.warning(
                'NSL value cannot be calculated without benchmark results'
            )
            return None

        required_columns = {'tokenizer', 'tokens_count'}
        if not required_columns.issubset(benchmark_results.columns):
            logger.warning(
                'NSL value cannot be calculated because required columns are missing'
            )
            return None

        data_custom = benchmark_results[
            benchmark_results['tokenizer'].str.contains(
                'custom tokenizer', case=False, na=False
            )
        ].copy()

        if data_custom.empty:
            logger.warning(
                'NSL value cannot be calculated without a custom tokenizer as reference'
            )
            return None

        data = []
        names = benchmark_results['tokenizer'].dropna().unique().tolist()
        for tok in tqdm(names):
            logger.info(
                f'NSL value is calculated for {tok} versus custom tokenizers'
            )
            data_chunk = benchmark_results[
                benchmark_results['tokenizer'] == tok
            ].copy()
            if data_chunk.empty:
                continue

            min_length = min(len(data_custom), len(data_chunk))
            if min_length == 0:
                continue

            ratios = [
                (x / y) if y else 0
                for x, y in zip(
                    data_custom['tokens_count'].to_numpy(dtype=float)[:min_length],
                    data_chunk['tokens_count'].to_numpy(dtype=float)[:min_length],
                )
            ]

            data_chunk = data_chunk.iloc[:min_length].copy()
            data_chunk['NSL'] = ratios
            data.append(data_chunk)

        if not data:
            return None

        data_NSL = pd.concat(data, ignore_index=True)

        return data_NSL


# [TOKENIZERS EXPLORER]



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
