from __future__ import annotations

from typing import Any


###############################################################################
DATASET_METRIC_CATALOG: list[dict[str, Any]] = [
    {
        "category_key": "corpus_scale",
        "category_label": "Corpus Scale & Structure",
        "metrics": [
            {"key": "corpus.document_count", "label": "Document count", "scope": "aggregate"},
            {"key": "corpus.total_chars", "label": "Total characters", "scope": "aggregate"},
            {"key": "corpus.total_words", "label": "Total words", "scope": "aggregate"},
            {"key": "corpus.unique_words", "label": "Vocabulary size", "scope": "aggregate", "core": True},
            {"key": "corpus.ttr", "label": "Type-token ratio", "scope": "aggregate"},
            {"key": "corpus.mattr", "label": "Moving-average TTR", "scope": "aggregate", "core": True},
            {"key": "doc.length_mean", "label": "Document length mean", "scope": "aggregate"},
            {"key": "doc.length_median", "label": "Document length median", "scope": "aggregate"},
            {"key": "doc.length_min", "label": "Document length min", "scope": "aggregate"},
            {"key": "doc.length_max", "label": "Document length max", "scope": "aggregate"},
            {"key": "doc.length_cv", "label": "Document length CV", "scope": "aggregate", "core": True},
            {"key": "doc.length_p10", "label": "Document length p10", "scope": "aggregate"},
            {"key": "doc.length_p25", "label": "Document length p25", "scope": "aggregate"},
            {"key": "doc.length_p50", "label": "Document length p50", "scope": "aggregate", "core": True},
            {"key": "doc.length_p75", "label": "Document length p75", "scope": "aggregate"},
            {"key": "doc.length_p90", "label": "Document length p90", "scope": "aggregate", "core": True},
            {"key": "doc.length_p95", "label": "Document length p95", "scope": "aggregate"},
            {"key": "doc.length_p99", "label": "Document length p99", "scope": "aggregate", "core": True},
            {"key": "doc.length_skewness", "label": "Document length skewness", "scope": "aggregate"},
            {"key": "doc.length_kurtosis", "label": "Document length kurtosis", "scope": "aggregate"},
            {"key": "doc.length_gini", "label": "Document length gini", "scope": "aggregate"},
            {"key": "doc.length_iqr", "label": "Document length IQR", "scope": "aggregate"},
            {"key": "doc.length_mad", "label": "Document length MAD", "scope": "aggregate"},
            {"key": "hist.document_length", "label": "Document length histogram", "scope": "aggregate", "value_kind": "histogram"},
            {"key": "hist.word_length", "label": "Word length histogram", "scope": "aggregate", "value_kind": "histogram"},
            {"key": "doc.length_chars", "label": "Document length (chars)", "scope": "per_document"},
            {"key": "doc.word_count", "label": "Document word count", "scope": "per_document"},
            {"key": "doc.avg_word_length", "label": "Document average word length", "scope": "per_document"},
            {"key": "doc.std_word_length", "label": "Document word length std", "scope": "per_document"},
        ],
    },
    {
        "category_key": "lexical_diversity",
        "category_label": "Lexical Diversity & Frequency",
        "metrics": [
            {"key": "words.shannon_entropy", "label": "Shannon entropy", "scope": "aggregate", "core": True},
            {"key": "words.normalized_entropy", "label": "Normalized entropy", "scope": "aggregate"},
            {"key": "words.zipf_slope", "label": "Zipf slope", "scope": "aggregate", "core": True},
            {"key": "words.hapax_ratio", "label": "Hapax ratio", "scope": "aggregate", "core": True},
            {"key": "words.dis_legomena_ratio", "label": "Dis legomena ratio", "scope": "aggregate"},
            {"key": "words.rare_tail_mass", "label": "Rare tail mass", "scope": "aggregate"},
            {"key": "words.topk_concentration", "label": "Top-k concentration", "scope": "aggregate"},
            {"key": "words.frequency_gini", "label": "Word frequency gini", "scope": "aggregate", "core": True},
            {"key": "words.hhi", "label": "HHI index", "scope": "aggregate"},
            {"key": "words.most_common", "label": "Most common words", "scope": "aggregate", "value_kind": "json"},
            {"key": "words.least_common", "label": "Least common words", "scope": "aggregate", "value_kind": "json"},
            {"key": "words.longest", "label": "Longest words", "scope": "aggregate", "value_kind": "json"},
            {"key": "words.shortest", "label": "Shortest words", "scope": "aggregate", "value_kind": "json"},
            {"key": "words.word_cloud", "label": "Word cloud terms", "scope": "aggregate", "value_kind": "json"},
            {"key": "words.zipf_curve", "label": "Zipf curve", "scope": "aggregate", "value_kind": "json"},
        ],
    },
    {
        "category_key": "word_character_signals",
        "category_label": "Word & Character Signals",
        "metrics": [
            {"key": "words.length_mean", "label": "Word length mean", "scope": "aggregate"},
            {"key": "words.length_median", "label": "Word length median", "scope": "aggregate"},
            {"key": "words.length_std", "label": "Word length std", "scope": "aggregate"},
            {"key": "chars.entropy", "label": "Character entropy", "scope": "aggregate"},
            {"key": "chars.whitespace_ratio", "label": "Whitespace ratio", "scope": "aggregate"},
            {"key": "chars.punctuation_ratio", "label": "Punctuation ratio", "scope": "aggregate"},
            {"key": "chars.digit_ratio", "label": "Digit ratio", "scope": "aggregate"},
            {"key": "chars.uppercase_ratio", "label": "Uppercase ratio", "scope": "aggregate"},
            {"key": "chars.non_ascii_ratio", "label": "Non-ASCII ratio", "scope": "aggregate"},
            {"key": "chars.control_ratio", "label": "Control character ratio", "scope": "aggregate"},
            {"key": "chars.other_ratio", "label": "Other character ratio", "scope": "aggregate"},
        ],
    },
    {
        "category_key": "document_quality",
        "category_label": "Document Quality Signals",
        "metrics": [
            {"key": "quality.empty_rate", "label": "Empty document rate", "scope": "aggregate"},
            {"key": "quality.near_empty_rate", "label": "Near-empty document rate", "scope": "aggregate"},
            {"key": "quality.exact_duplicate_rate", "label": "Exact duplicate rate", "scope": "aggregate", "core": True},
            {"key": "quality.duplicate_rate", "label": "Duplicate rate", "scope": "aggregate", "core": True},
            {"key": "quality.near_duplicate_rate", "label": "Near-duplicate rate", "scope": "aggregate"},
            {"key": "quality.language_consistency", "label": "Language consistency score", "scope": "aggregate"},
            {"key": "quality.avg_sentence_count", "label": "Average sentence count", "scope": "aggregate"},
            {"key": "quality.avg_sentence_length", "label": "Average sentence length", "scope": "aggregate"},
            {"key": "quality.sentence_length_variance", "label": "Sentence length variance", "scope": "aggregate"},
            {"key": "quality.is_exact_duplicate", "label": "Is exact duplicate", "scope": "per_document"},
            {"key": "quality.is_near_duplicate", "label": "Is near duplicate", "scope": "per_document"},
            {"key": "quality.language_tag", "label": "Language tag", "scope": "per_document", "value_kind": "text"},
        ],
    },
    {
        "category_key": "structural_regularity",
        "category_label": "Structural Regularity",
        "metrics": [
            {"key": "structure.avg_paragraph_count", "label": "Average paragraph count", "scope": "aggregate"},
            {"key": "structure.line_break_density", "label": "Line-break density", "scope": "aggregate"},
            {"key": "structure.html_tag_ratio", "label": "HTML tag ratio", "scope": "aggregate"},
            {"key": "structure.url_density", "label": "URL density", "scope": "aggregate"},
            {"key": "structure.email_density", "label": "Email density", "scope": "aggregate"},
        ],
    },
    {
        "category_key": "compression_redundancy",
        "category_label": "Compression & Redundancy",
        "metrics": [
            {"key": "compression.ratio", "label": "Compression ratio", "scope": "aggregate", "core": True},
            {"key": "compression.chars_per_unique_word", "label": "Characters per unique word", "scope": "aggregate"},
            {"key": "compression.avg_repetition_factor", "label": "Average repetition factor", "scope": "aggregate"},
            {"key": "compression.bigram_repetition_rate", "label": "Bigram repetition rate", "scope": "aggregate"},
            {"key": "doc.compression_ratio", "label": "Document compression ratio", "scope": "per_document"},
        ],
    },
]


###############################################################################
def flatten_metric_keys(metric_catalog: list[dict[str, Any]] | None = None) -> list[str]:
    catalog = metric_catalog or DATASET_METRIC_CATALOG
    keys: list[str] = []
    for category in catalog:
        metrics = category.get("metrics")
        if not isinstance(metrics, list):
            continue
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            key = metric.get("key")
            if isinstance(key, str) and key:
                keys.append(key)
    return keys


###############################################################################
def default_selected_metric_keys() -> list[str]:
    keys: list[str] = []
    for category in DATASET_METRIC_CATALOG:
        metrics = category.get("metrics")
        if not isinstance(metrics, list):
            continue
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            key = metric.get("key")
            if not isinstance(key, str) or not key:
                continue
            if bool(metric.get("core")) or str(metric.get("scope", "aggregate")) == "aggregate":
                keys.append(key)
    return sorted(set(keys))
