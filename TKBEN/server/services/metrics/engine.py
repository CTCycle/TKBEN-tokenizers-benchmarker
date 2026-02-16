from __future__ import annotations

import gzip
import hashlib
import math
import re
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from TKBEN.server.services.metrics.duplicates import SimHashNearDuplicateAnalyzer
from TKBEN.server.services.metrics.frequencies import DiskBackedFrequencyStore

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)

STOPWORDS: dict[str, set[str]] = {
    "en": {"the", "and", "of", "to", "a", "in", "is", "that", "for", "on", "it", "with"},
    "es": {"el", "la", "de", "que", "y", "en", "a", "los", "se", "del", "las", "por"},
    "fr": {"le", "de", "un", "et", "la", "les", "des", "en", "du", "est", "pour", "que"},
    "de": {"der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des", "auf"},
    "it": {"di", "e", "il", "la", "che", "a", "per", "in", "un", "del", "le", "si"},
}


###############################################################################
@dataclass
class RunningMoments:
    count: int = 0
    sum_value: float = 0.0
    sum_squared: float = 0.0
    sum_cubed: float = 0.0
    sum_fourth: float = 0.0

    # -------------------------------------------------------------------------
    def update(self, value: float) -> None:
        self.count += 1
        self.sum_value += value
        self.sum_squared += value * value
        self.sum_cubed += value * value * value
        self.sum_fourth += value * value * value * value

    # -------------------------------------------------------------------------
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum_value / float(self.count)

    # -------------------------------------------------------------------------
    def variance(self) -> float:
        if self.count == 0:
            return 0.0
        mu = self.mean()
        return max(0.0, (self.sum_squared / float(self.count)) - (mu * mu))

    # -------------------------------------------------------------------------
    def std(self) -> float:
        return math.sqrt(self.variance())

    # -------------------------------------------------------------------------
    def skewness(self) -> float:
        if self.count == 0:
            return 0.0
        mu = self.mean()
        var = self.variance()
        if var <= 0.0:
            return 0.0
        m3 = (self.sum_cubed / float(self.count)) - (3 * mu * var) - (mu ** 3)
        return m3 / (var ** 1.5)

    # -------------------------------------------------------------------------
    def kurtosis(self) -> float:
        if self.count == 0:
            return 0.0
        mu = self.mean()
        var = self.variance()
        if var <= 0.0:
            return 0.0
        m4 = (
            (self.sum_fourth / float(self.count))
            - (4 * mu * (self.sum_cubed / float(self.count)))
            + (6 * (mu ** 2) * (self.sum_squared / float(self.count)))
            - (3 * (mu ** 4))
        )
        return (m4 / (var * var)) - 3.0


###############################################################################
class RollingMattr:
    def __init__(self, window_size: int) -> None:
        self.window_size = max(10, int(window_size))
        self.window: deque[str] = deque()
        self.counts: Counter[str] = Counter()
        self.total_windows = 0
        self.sum_ttr = 0.0

    # -------------------------------------------------------------------------
    def update(self, token: str) -> None:
        self.window.append(token)
        self.counts[token] += 1
        if len(self.window) > self.window_size:
            removed = self.window.popleft()
            self.counts[removed] -= 1
            if self.counts[removed] <= 0:
                del self.counts[removed]
        if len(self.window) == self.window_size:
            self.total_windows += 1
            self.sum_ttr += len(self.counts) / float(self.window_size)

    # -------------------------------------------------------------------------
    def value(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return self.sum_ttr / float(self.total_windows)


###############################################################################
class DatasetMetricsEngine:
    def __init__(self, parameters: dict[str, Any]) -> None:
        self.parameters = parameters
        self.word_frequency = DiskBackedFrequencyStore(
            memory_limit=int(parameters.get("max_vocab_in_memory", 200_000))
        )
        self.mattr = RollingMattr(int(parameters.get("mattr_window", 100)))
        self.near_duplicate = SimHashNearDuplicateAnalyzer(
            similarity_threshold=float(parameters.get("near_duplicate_threshold", 0.9)),
            bands=int(parameters.get("simhash_bands", 4)),
            bits=64,
        )

        self.document_count = 0
        self.total_chars = 0
        self.total_words = 0
        self.document_length_counts: Counter[int] = Counter()
        self.word_length_counts: Counter[int] = Counter()
        self.document_length_moments = RunningMoments()
        self.word_length_moments = RunningMoments()
        self.char_frequency: Counter[str] = Counter()

        self.empty_documents = 0
        self.near_empty_documents = 0
        self.exact_duplicate_documents = 0
        self.near_duplicate_documents = 0
        self.exact_hashes: Counter[str] = Counter()

        self.language_tag_counts: Counter[str] = Counter()
        self.total_sentences = 0
        self.total_sentence_words = 0
        self.sentence_length_moments = RunningMoments()
        self.total_paragraphs = 0
        self.total_line_breaks = 0
        self.total_html_tag_chars = 0
        self.total_urls = 0
        self.total_emails = 0

        self.compressed_bytes = 0
        self.raw_bytes = 0
        self.bigram_repeated = 0
        self.bigram_total = 0

        self.whitespace_chars = 0
        self.punctuation_chars = 0
        self.digit_chars = 0
        self.uppercase_chars = 0
        self.non_ascii_chars = 0
        self.control_chars = 0

    # -------------------------------------------------------------------------
    def _detect_language_tag(self, tokens: list[str], text: str) -> str:
        sample = [token.lower() for token in tokens[:250]]
        lang_scores: dict[str, int] = {}
        for lang, stopwords in STOPWORDS.items():
            hits = sum(1 for token in sample if token in stopwords)
            lang_scores[lang] = hits
        best_lang = max(lang_scores, key=lang_scores.get) if lang_scores else "unknown"
        if lang_scores.get(best_lang, 0) >= 2:
            return best_lang

        script_counts: Counter[str] = Counter()
        for char in text:
            if not char.isalpha():
                continue
            codepoint = ord(char)
            if 0x0400 <= codepoint <= 0x052F:
                script_counts["cyrillic"] += 1
            elif 0x0600 <= codepoint <= 0x06FF:
                script_counts["arabic"] += 1
            elif 0x4E00 <= codepoint <= 0x9FFF:
                script_counts["han"] += 1
            elif 0x0370 <= codepoint <= 0x03FF:
                script_counts["greek"] += 1
            else:
                script_counts["latin"] += 1
        if not script_counts:
            return "unknown"
        return script_counts.most_common(1)[0][0]

    # -------------------------------------------------------------------------
    def _extract_sentences(self, text: str) -> list[str]:
        parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text)]
        return [part for part in parts if part]

    # -------------------------------------------------------------------------
    def _compute_doc_compression_ratio(self, text: str) -> float:
        raw = text.encode("utf-8", errors="ignore")
        if not raw:
            return 0.0
        compressed = gzip.compress(raw)
        self.raw_bytes += len(raw)
        self.compressed_bytes += len(compressed)
        return len(compressed) / float(len(raw))

    # -------------------------------------------------------------------------
    def _mean_word_length(self, words: list[str]) -> float:
        if not words:
            return 0.0
        return sum(len(word) for word in words) / float(len(words))

    # -------------------------------------------------------------------------
    def _std_word_length(self, words: list[str], mean_word_length: float) -> float:
        if not words:
            return 0.0
        variance = sum((len(word) - mean_word_length) ** 2 for word in words) / float(len(words))
        return math.sqrt(max(0.0, variance))

    # -------------------------------------------------------------------------
    def process_document(self, document_id: int, text: str) -> list[dict[str, Any]]:
        content = text if isinstance(text, str) else str(text or "")
        doc_length = len(content)
        words = WORD_PATTERN.findall(content.lower())
        word_count = len(words)

        self.document_count += 1
        self.total_chars += doc_length
        self.total_words += word_count
        self.document_length_counts[doc_length] += 1
        self.document_length_moments.update(float(doc_length))

        if word_count == 0:
            self.empty_documents += 1
        if word_count <= int(self.parameters.get("near_empty_threshold_words", 3)):
            self.near_empty_documents += 1

        if words:
            self.word_frequency.add_many(words)
            for token in words:
                self.mattr.update(token)
                length = len(token)
                self.word_length_counts[length] += 1
                self.word_length_moments.update(float(length))

        digest = hashlib.blake2b(content.encode("utf-8", errors="ignore"), digest_size=16).hexdigest()
        self.exact_hashes[digest] += 1
        is_exact_duplicate = self.exact_hashes[digest] > 1
        if is_exact_duplicate:
            self.exact_duplicate_documents += 1

        is_near_duplicate = self.near_duplicate.check_and_add(words)
        if is_near_duplicate:
            self.near_duplicate_documents += 1

        doc_compression_ratio = self._compute_doc_compression_ratio(content)

        for char in content:
            self.char_frequency[char] += 1
            if char.isspace():
                self.whitespace_chars += 1
            if char.isdigit():
                self.digit_chars += 1
            if char.isupper():
                self.uppercase_chars += 1
            if ord(char) > 127:
                self.non_ascii_chars += 1
            category = unicodedata.category(char)
            if category.startswith("P"):
                self.punctuation_chars += 1
            if category.startswith("C"):
                self.control_chars += 1

        language_tag = self._detect_language_tag(words, content)
        self.language_tag_counts[language_tag] += 1

        sentences = self._extract_sentences(content)
        self.total_sentences += len(sentences)
        for sentence in sentences:
            sentence_words = len(WORD_PATTERN.findall(sentence))
            self.total_sentence_words += sentence_words
            self.sentence_length_moments.update(float(sentence_words))

        paragraph_count = len([segment for segment in re.split(r"\n\s*\n", content) if segment.strip()])
        self.total_paragraphs += paragraph_count
        self.total_line_breaks += content.count("\n")

        html_tags = HTML_TAG_PATTERN.findall(content)
        self.total_html_tag_chars += sum(len(tag) for tag in html_tags)
        self.total_urls += len(URL_PATTERN.findall(content))
        self.total_emails += len(EMAIL_PATTERN.findall(content))

        if word_count >= 2:
            bigrams = Counter(zip(words, words[1:]))
            self.bigram_total += max(0, word_count - 1)
            self.bigram_repeated += sum(max(0, value - 1) for value in bigrams.values())

        avg_word_length = self._mean_word_length(words)
        std_word_length = self._std_word_length(words, avg_word_length)
        return [
            {"metric_key": "doc.length_chars", "document_id": document_id, "numeric_value": float(doc_length)},
            {"metric_key": "doc.word_count", "document_id": document_id, "numeric_value": float(word_count)},
            {"metric_key": "doc.avg_word_length", "document_id": document_id, "numeric_value": float(avg_word_length)},
            {"metric_key": "doc.std_word_length", "document_id": document_id, "numeric_value": float(std_word_length)},
            {"metric_key": "doc.compression_ratio", "document_id": document_id, "numeric_value": float(doc_compression_ratio)},
            {"metric_key": "quality.is_exact_duplicate", "document_id": document_id, "numeric_value": 1.0 if is_exact_duplicate else 0.0},
            {"metric_key": "quality.is_near_duplicate", "document_id": document_id, "numeric_value": 1.0 if is_near_duplicate else 0.0},
            {"metric_key": "quality.language_tag", "document_id": document_id, "text_value": language_tag},
        ]

    # -------------------------------------------------------------------------
    def _percentile(self, counts: Counter[int], percentile: float) -> float:
        if not counts:
            return 0.0
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        rank = max(1, int(math.ceil((percentile / 100.0) * total)))
        cumulative = 0
        for value in sorted(counts.keys()):
            cumulative += counts[value]
            if cumulative >= rank:
                return float(value)
        return float(max(counts.keys()))

    # -------------------------------------------------------------------------
    def _median_absolute_deviation(self, counts: Counter[int]) -> float:
        if not counts:
            return 0.0
        median = self._percentile(counts, 50.0)
        deviations: Counter[int] = Counter()
        for value, count in counts.items():
            deviations[int(abs(value - median))] += int(count)
        return float(self._percentile(deviations, 50.0))

    # -------------------------------------------------------------------------
    def _weighted_gini(self, counts: Counter[int]) -> float:
        if not counts:
            return 0.0
        total_count = sum(counts.values())
        total_value = sum(value * count for value, count in counts.items())
        if total_count <= 0 or total_value <= 0:
            return 0.0
        cumulative = 0
        numerator = 0.0
        for value in sorted(counts.keys()):
            count = int(counts[value])
            numerator += value * count * (2 * cumulative + count - total_count)
            cumulative += count
        return numerator / float(total_count * total_value)

    # -------------------------------------------------------------------------
    def _build_histogram(self, counts: Counter[int], bins: int) -> dict[str, Any]:
        if not counts:
            return {
                "bins": [],
                "counts": [],
                "bin_edges": [],
                "min_length": 0,
                "max_length": 0,
                "mean_length": 0.0,
                "median_length": 0.0,
            }
        min_value = min(counts.keys())
        max_value = max(counts.keys())
        total = sum(counts.values())
        bin_count = max(1, int(bins))
        width = max(1, int(math.ceil(((max_value - min_value) + 1) / float(bin_count))))
        edges = [min_value + (index * width) for index in range(bin_count + 1)]
        histogram_counts = [0] * bin_count
        for value, count in counts.items():
            index = min(bin_count - 1, int((value - min_value) // width))
            histogram_counts[index] += int(count)
        labels = []
        for index in range(bin_count):
            left = edges[index]
            right = edges[index + 1] - 1
            labels.append(f"{left}-{max(left, right)}")
        mean_value = sum(value * count for value, count in counts.items()) / float(total)
        return {
            "bins": labels,
            "counts": histogram_counts,
            "bin_edges": [float(edge) for edge in edges],
            "min_length": int(min_value),
            "max_length": int(max_value),
            "mean_length": float(mean_value),
            "median_length": float(self._percentile(counts, 50.0)),
        }

    # -------------------------------------------------------------------------
    def _shannon_entropy(self) -> float:
        total = self.word_frequency.total_count()
        if total <= 0:
            return 0.0
        entropy = 0.0
        for _, count in self.word_frequency.iter_counts():
            probability = count / float(total)
            if probability > 0.0:
                entropy -= probability * math.log2(probability)
        return entropy

    # -------------------------------------------------------------------------
    def _char_entropy(self) -> float:
        total = sum(self.char_frequency.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for count in self.char_frequency.values():
            probability = count / float(total)
            if probability > 0.0:
                entropy -= probability * math.log2(probability)
        return entropy

    # -------------------------------------------------------------------------
    def _zipf_slope_and_curve(self) -> tuple[float, list[dict[str, Any]]]:
        rank = 0
        n = 0
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0
        curve: list[dict[str, Any]] = []
        for token, count in self.word_frequency.iter_sorted_counts(descending=True):
            rank += 1
            log_rank = math.log(float(rank))
            log_freq = math.log(float(max(1, count)))
            n += 1
            sum_x += log_rank
            sum_y += log_freq
            sum_xy += log_rank * log_freq
            sum_x2 += log_rank * log_rank
            if rank <= 200:
                curve.append({"rank": rank, "frequency": int(count), "word": token})
        if n < 2:
            return 0.0, curve
        denominator = (n * sum_x2) - (sum_x * sum_x)
        if denominator == 0.0:
            return 0.0, curve
        slope = ((n * sum_xy) - (sum_x * sum_y)) / denominator
        return float(slope), curve

    # -------------------------------------------------------------------------
    def _top_k_concentration(self, top_k: int) -> float:
        total_words = self.word_frequency.total_count()
        if total_words <= 0:
            return 0.0
        top_sum = self.word_frequency.sum_top_k(top_k)
        return top_sum / float(total_words)

    # -------------------------------------------------------------------------
    def _rare_tail_mass(self, bottom_fraction: float) -> float:
        total_words = self.word_frequency.total_count()
        unique_words = self.word_frequency.unique_count()
        if total_words <= 0 or unique_words <= 0:
            return 0.0
        bottom_k = max(1, int(math.ceil(unique_words * max(0.0, min(1.0, bottom_fraction)))))
        bottom_sum = self.word_frequency.sum_bottom_k(bottom_k)
        return bottom_sum / float(total_words)

    # -------------------------------------------------------------------------
    def _word_frequency_gini(self) -> float:
        counts = Counter()
        for _, count in self.word_frequency.iter_counts():
            counts[int(count)] += 1
        return self._weighted_gini(counts)

    # -------------------------------------------------------------------------
    def _word_frequency_hhi(self) -> float:
        total_words = self.word_frequency.total_count()
        if total_words <= 0:
            return 0.0
        hhi = 0.0
        for _, count in self.word_frequency.iter_counts():
            share = count / float(total_words)
            hhi += share * share
        return hhi

    # -------------------------------------------------------------------------
    def _common_word_payload(self, top_k: int = 10, least: bool = False) -> list[dict[str, Any]]:
        items = self.word_frequency.bottom_k(top_k) if least else self.word_frequency.top_k(top_k)
        return [{"word": token, "count": int(count)} for token, count in items]

    # -------------------------------------------------------------------------
    def _word_items_for_length(self, ascending: bool, top_k: int = 15) -> list[dict[str, Any]]:
        target_k = max(1, int(top_k))
        rows = (
            self.word_frequency.shortest_k(target_k)
            if ascending
            else self.word_frequency.longest_k(target_k)
        )
        return [
            {"word": token, "length": len(token), "count": int(count)}
            for token, count in rows
        ]

    # -------------------------------------------------------------------------
    def _word_cloud_payload(self, top_k: int = 120) -> list[dict[str, Any]]:
        items = self.word_frequency.top_k(top_k)
        if not items:
            return []
        max_count = max(count for _, count in items)
        if max_count <= 0:
            return []
        return [
            {
                "word": token,
                "count": int(count),
                "weight": max(1, int(round((count / float(max_count)) * 100))),
            }
            for token, count in items
        ]

    # -------------------------------------------------------------------------
    def finalize(self, histogram_bins: int) -> dict[str, Any]:
        total_chars = max(0, self.total_chars)
        total_words = max(0, self.total_words)
        vocabulary_size = self.word_frequency.unique_count()
        shannon_entropy = self._shannon_entropy()
        normalized_entropy = (
            shannon_entropy / math.log2(vocabulary_size)
            if vocabulary_size > 1
            else 0.0
        )
        zipf_slope, zipf_curve = self._zipf_slope_and_curve()

        hapax = self.word_frequency.count_frequency_of_frequency(1)
        dis_legomena = self.word_frequency.count_frequency_of_frequency(2)
        hapax_ratio = (hapax / float(vocabulary_size)) if vocabulary_size > 0 else 0.0
        dis_legomena_ratio = (
            dis_legomena / float(vocabulary_size) if vocabulary_size > 0 else 0.0
        )
        rare_tail_mass = self._rare_tail_mass(float(self.parameters.get("rare_tail_percent", 0.10)))
        topk_concentration = self._top_k_concentration(int(self.parameters.get("top_k_concentration", 20)))
        word_frequency_gini = self._word_frequency_gini()
        hhi = self._word_frequency_hhi()

        doc_mean = self.document_length_moments.mean()
        doc_std = self.document_length_moments.std()
        doc_cv = (doc_std / doc_mean) if doc_mean > 0 else 0.0
        p10 = self._percentile(self.document_length_counts, 10.0)
        p25 = self._percentile(self.document_length_counts, 25.0)
        p50 = self._percentile(self.document_length_counts, 50.0)
        p75 = self._percentile(self.document_length_counts, 75.0)
        p90 = self._percentile(self.document_length_counts, 90.0)
        p95 = self._percentile(self.document_length_counts, 95.0)
        p99 = self._percentile(self.document_length_counts, 99.0)

        language_consistency = 0.0
        if self.document_count > 0 and self.language_tag_counts:
            language_consistency = self.language_tag_counts.most_common(1)[0][1] / float(self.document_count)

        compression_ratio = (
            self.compressed_bytes / float(self.raw_bytes) if self.raw_bytes > 0 else 0.0
        )
        chars_per_unique_word = (
            total_chars / float(vocabulary_size) if vocabulary_size > 0 else 0.0
        )
        avg_repetition_factor = (
            total_words / float(vocabulary_size) if vocabulary_size > 0 else 0.0
        )
        bigram_repetition_rate = (
            self.bigram_repeated / float(self.bigram_total) if self.bigram_total > 0 else 0.0
        )

        sentence_length_variance = self.sentence_length_moments.variance()
        avg_sentence_count = (
            self.total_sentences / float(self.document_count) if self.document_count > 0 else 0.0
        )
        avg_sentence_length = (
            self.total_sentence_words / float(self.total_sentences) if self.total_sentences > 0 else 0.0
        )

        doc_hist = self._build_histogram(self.document_length_counts, bins=histogram_bins)
        word_hist = self._build_histogram(self.word_length_counts, bins=histogram_bins)

        char_entropy = self._char_entropy()
        denominator_chars = float(total_chars) if total_chars > 0 else 1.0
        whitespace_ratio = self.whitespace_chars / denominator_chars
        punctuation_ratio = self.punctuation_chars / denominator_chars
        digit_ratio = self.digit_chars / denominator_chars
        uppercase_ratio = self.uppercase_chars / denominator_chars
        non_ascii_ratio = self.non_ascii_chars / denominator_chars
        control_ratio = self.control_chars / denominator_chars
        other_ratio = max(
            0.0,
            1.0
            - whitespace_ratio
            - punctuation_ratio
            - digit_ratio
            - uppercase_ratio
            - non_ascii_ratio
            - control_ratio,
        )

        metric_rows: list[dict[str, Any]] = [
            {"metric_key": "corpus.document_count", "numeric_value": float(self.document_count)},
            {"metric_key": "corpus.total_chars", "numeric_value": float(total_chars)},
            {"metric_key": "corpus.total_words", "numeric_value": float(total_words)},
            {"metric_key": "corpus.unique_words", "numeric_value": float(vocabulary_size)},
            {"metric_key": "corpus.ttr", "numeric_value": (vocabulary_size / float(total_words)) if total_words > 0 else 0.0},
            {"metric_key": "corpus.mattr", "numeric_value": self.mattr.value()},
            {"metric_key": "doc.length_mean", "numeric_value": doc_mean},
            {"metric_key": "doc.length_median", "numeric_value": p50},
            {"metric_key": "doc.length_min", "numeric_value": float(min(self.document_length_counts.keys(), default=0))},
            {"metric_key": "doc.length_max", "numeric_value": float(max(self.document_length_counts.keys(), default=0))},
            {"metric_key": "doc.length_cv", "numeric_value": doc_cv},
            {"metric_key": "doc.length_p10", "numeric_value": p10},
            {"metric_key": "doc.length_p25", "numeric_value": p25},
            {"metric_key": "doc.length_p50", "numeric_value": p50},
            {"metric_key": "doc.length_p75", "numeric_value": p75},
            {"metric_key": "doc.length_p90", "numeric_value": p90},
            {"metric_key": "doc.length_p95", "numeric_value": p95},
            {"metric_key": "doc.length_p99", "numeric_value": p99},
            {"metric_key": "doc.length_skewness", "numeric_value": self.document_length_moments.skewness()},
            {"metric_key": "doc.length_kurtosis", "numeric_value": self.document_length_moments.kurtosis()},
            {"metric_key": "doc.length_gini", "numeric_value": self._weighted_gini(self.document_length_counts)},
            {"metric_key": "doc.length_iqr", "numeric_value": p75 - p25},
            {"metric_key": "doc.length_mad", "numeric_value": self._median_absolute_deviation(self.document_length_counts)},
            {"metric_key": "words.shannon_entropy", "numeric_value": shannon_entropy},
            {"metric_key": "words.normalized_entropy", "numeric_value": normalized_entropy},
            {"metric_key": "words.zipf_slope", "numeric_value": zipf_slope},
            {"metric_key": "words.hapax_ratio", "numeric_value": hapax_ratio},
            {"metric_key": "words.dis_legomena_ratio", "numeric_value": dis_legomena_ratio},
            {"metric_key": "words.rare_tail_mass", "numeric_value": rare_tail_mass},
            {"metric_key": "words.topk_concentration", "numeric_value": topk_concentration},
            {"metric_key": "words.frequency_gini", "numeric_value": word_frequency_gini},
            {"metric_key": "words.hhi", "numeric_value": hhi},
            {"metric_key": "words.length_mean", "numeric_value": self.word_length_moments.mean()},
            {"metric_key": "words.length_median", "numeric_value": self._percentile(self.word_length_counts, 50.0)},
            {"metric_key": "words.length_std", "numeric_value": self.word_length_moments.std()},
            {"metric_key": "chars.entropy", "numeric_value": char_entropy},
            {"metric_key": "chars.whitespace_ratio", "numeric_value": whitespace_ratio},
            {"metric_key": "chars.punctuation_ratio", "numeric_value": punctuation_ratio},
            {"metric_key": "chars.digit_ratio", "numeric_value": digit_ratio},
            {"metric_key": "chars.uppercase_ratio", "numeric_value": uppercase_ratio},
            {"metric_key": "chars.non_ascii_ratio", "numeric_value": non_ascii_ratio},
            {"metric_key": "chars.control_ratio", "numeric_value": control_ratio},
            {"metric_key": "chars.other_ratio", "numeric_value": other_ratio},
            {"metric_key": "quality.empty_rate", "numeric_value": (self.empty_documents / float(self.document_count)) if self.document_count > 0 else 0.0},
            {"metric_key": "quality.near_empty_rate", "numeric_value": (self.near_empty_documents / float(self.document_count)) if self.document_count > 0 else 0.0},
            {"metric_key": "quality.exact_duplicate_rate", "numeric_value": (self.exact_duplicate_documents / float(self.document_count)) if self.document_count > 0 else 0.0},
            {"metric_key": "quality.duplicate_rate", "numeric_value": (self.exact_duplicate_documents / float(self.document_count)) if self.document_count > 0 else 0.0},
            {"metric_key": "quality.near_duplicate_rate", "numeric_value": (self.near_duplicate_documents / float(self.document_count)) if self.document_count > 0 else 0.0},
            {"metric_key": "quality.language_consistency", "numeric_value": language_consistency},
            {"metric_key": "quality.avg_sentence_count", "numeric_value": avg_sentence_count},
            {"metric_key": "quality.avg_sentence_length", "numeric_value": avg_sentence_length},
            {"metric_key": "quality.sentence_length_variance", "numeric_value": sentence_length_variance},
            {"metric_key": "structure.avg_paragraph_count", "numeric_value": (self.total_paragraphs / float(self.document_count)) if self.document_count > 0 else 0.0},
            {"metric_key": "structure.line_break_density", "numeric_value": self.total_line_breaks / denominator_chars},
            {"metric_key": "structure.html_tag_ratio", "numeric_value": self.total_html_tag_chars / denominator_chars},
            {"metric_key": "structure.url_density", "numeric_value": self.total_urls / float(max(1, total_words))},
            {"metric_key": "structure.email_density", "numeric_value": self.total_emails / float(max(1, total_words))},
            {"metric_key": "compression.ratio", "numeric_value": compression_ratio},
            {"metric_key": "compression.chars_per_unique_word", "numeric_value": chars_per_unique_word},
            {"metric_key": "compression.avg_repetition_factor", "numeric_value": avg_repetition_factor},
            {"metric_key": "compression.bigram_repetition_rate", "numeric_value": bigram_repetition_rate},
            {"metric_key": "words.most_common", "json_value": self._common_word_payload(10, least=False)},
            {"metric_key": "words.least_common", "json_value": self._common_word_payload(10, least=True)},
            {"metric_key": "words.longest", "json_value": self._word_items_for_length(ascending=False, top_k=15)},
            {"metric_key": "words.shortest", "json_value": self._word_items_for_length(ascending=True, top_k=15)},
            {"metric_key": "words.word_cloud", "json_value": self._word_cloud_payload(120)},
            {"metric_key": "words.zipf_curve", "json_value": zipf_curve},
        ]

        self.word_frequency.close()
        return {
            "metric_rows": metric_rows,
            "document_histogram": doc_hist,
            "word_histogram": word_hist,
        }
