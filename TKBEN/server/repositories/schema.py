from __future__ import annotations
from typing import Any

from sqlalchemy import (    
    Column,
    Float,    
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import TypeDecorator

Base = declarative_base()


###############################################################################
class JSONSequence(TypeDecorator):
    impl = JSON
    cache_ok = True

    def process_bind_param(self, value, dialect) -> Any | None:  # type: ignore[override]
        return value

    def process_result_value(self, value, dialect) -> Any | None:  # type: ignore[override]
        return value


###############################################################################
class IntSequence(JSONSequence):
    pass


###############################################################################
class TokenizationLocalStats(Base):
    __tablename__ = "TOKENIZATION_LOCAL_STATS"
    tokenizer = Column(String, primary_key=True)
    text = Column(String, primary_key=True)
    num_characters = Column(Integer)
    words_count = Column(Integer)
    AVG_words_length = Column(Float)
    tokens_count = Column(Integer)
    tokens_characters = Column(Integer)
    AVG_tokens_length = Column(Float)
    tokens_to_words_ratio = Column(Float)
    bytes_per_token = Column(Float)
    boundary_preservation_rate = Column(Float)
    round_trip_token_fidelity = Column(Float)
    round_trip_text_fidelity = Column(Float)
    determinism_stability = Column(Float)
    bytes_per_character = Column(Float)
    characters_per_token = Column(Float)
    token_length_variance = Column(Float)
    token_length_std = Column(Float)
    __table_args__ = (UniqueConstraint("tokenizer", "text"),)


###############################################################################
class TokenizationGlobalMetrics(Base):
    __tablename__ = "TOKENIZATION_GLOBAL_METRICS"
    tokenizer = Column(String, primary_key=True)
    dataset_name = Column(String, primary_key=True)
    tokenization_speed_tps = Column(Float)
    throughput_chars_per_sec = Column(Float)
    model_size_mb = Column(Float)
    vocabulary_size = Column(Integer)
    avg_sequence_length = Column(Float)
    median_sequence_length = Column(Float)
    subword_fertility = Column(Float)
    oov_rate = Column(Float)
    word_recovery_rate = Column(Float)
    character_coverage = Column(Float)
    segmentation_consistency = Column(Float)
    determinism_rate = Column(Float)
    token_distribution_entropy = Column(Float)
    rare_token_tail_1 = Column(Integer)
    rare_token_tail_2 = Column(Integer)
    boundary_preservation_rate = Column(Float)
    compression_chars_per_token = Column(Float)
    compression_bytes_per_character = Column(Float)
    round_trip_fidelity_rate = Column(Float)
    round_trip_text_fidelity_rate = Column(Float)
    token_id_ordering_monotonicity = Column(Float)
    token_unigram_coverage = Column(Float)
    token_length_variance = Column(Float)
    token_length_std = Column(Float)
    __table_args__ = (UniqueConstraint("tokenizer", "dataset_name"),)


###############################################################################
class VocabularyStatistics(Base):
    __tablename__ = "VOCABULARY_STATISTICS"
    tokenizer = Column(String, primary_key=True)
    number_tokens_from_vocabulary = Column(Integer)
    number_tokens_from_decode = Column(Integer)
    number_shared_tokens = Column(Integer)
    number_unshared_tokens = Column(Integer)
    percentage_subwords = Column(Float)
    percentage_true_words = Column(Float)
    __table_args__ = (UniqueConstraint("tokenizer"),)


###############################################################################
class Vocabulary(Base):
    __tablename__ = "VOCABULARY"
    tokenizer = Column(String, primary_key=True)
    token_id = Column(Integer, primary_key=True)
    vocabulary_tokens = Column(String)
    decoded_tokens = Column(String)
    __table_args__ = (UniqueConstraint("tokenizer", "token_id"),)


###############################################################################
class TextDataset(Base):
    __tablename__ = "TEXT_DATASET"
    dataset_name = Column(String, primary_key=True)
    text = Column(String, primary_key=True)
    __table_args__ = (UniqueConstraint("dataset_name", "text"),)


###############################################################################
class TextDatasetStatistics(Base):
    __tablename__ = "TEXT_DATASET_STATISTICS"
    dataset_name = Column(String, primary_key=True)
    text = Column(String, primary_key=True)
    words_count = Column(Integer)
    AVG_words_length = Column(Float)
    STD_words_length = Column(Float)
    __table_args__ = (UniqueConstraint("dataset_name", "text"),)


###############################################################################
class TextDatasetReports(Base):
    __tablename__ = "TEXT_DATASET_REPORTS"
    dataset_name = Column(String, primary_key=True)
    document_count = Column(Integer)
    document_bins = Column(JSONSequence)
    document_counts = Column(JSONSequence)
    document_bin_edges = Column(JSONSequence)
    document_min_length = Column(Integer)
    document_max_length = Column(Integer)
    document_mean_length = Column(Float)
    document_median_length = Column(Float)
    word_bins = Column(JSONSequence)
    word_counts = Column(JSONSequence)
    word_bin_edges = Column(JSONSequence)
    word_min_length = Column(Integer)
    word_max_length = Column(Integer)
    word_mean_length = Column(Float)
    word_median_length = Column(Float)
    most_common_words = Column(JSONSequence)
    least_common_words = Column(JSONSequence)
    __table_args__ = (UniqueConstraint("dataset_name"),)
