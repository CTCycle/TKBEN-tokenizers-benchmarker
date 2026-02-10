from __future__ import annotations

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

from TKBEN.server.repositories.schemas.types import JSONMapping, JSONSequence

Base = declarative_base()


###############################################################################
class TextDataset(Base):
    __tablename__ = "text_dataset"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    name = Column(String, nullable=False)
    text = Column(String, nullable=False)
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class TokenizationLocalStats(Base):
    __tablename__ = "tokenization_local_stats"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    tokenizer = Column(String, nullable=False)
    name = Column(String, nullable=False)
    text_id = Column(Integer, ForeignKey("text_dataset.id"), nullable=False)
    tokens_count = Column(Integer)
    tokens_to_words_ratio = Column(Float)
    bytes_per_token = Column(Float)
    boundary_preservation_rate = Column(Float)
    round_trip_token_fidelity = Column(Float)
    round_trip_text_fidelity = Column(Float)
    determinism_stability = Column(Float)
    bytes_per_character = Column(Float)
    __table_args__ = (UniqueConstraint("tokenizer", "text_id"),)


###############################################################################
class TokenizationGlobalStats(Base):
    __tablename__ = "tokenization_global_stats"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    tokenizer = Column(String, nullable=False)
    name = Column(String, nullable=False)
    tokenization_speed_tps = Column(Float)
    throughput_chars_per_sec = Column(Float)
    model_size_mb = Column(Float)
    vocabulary_size = Column(Integer)
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
    __table_args__ = (UniqueConstraint("tokenizer", "name"),)


###############################################################################
class VocabularyStatistics(Base):
    __tablename__ = "vocabulary_statistics"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    tokenizer = Column(String, nullable=False)
    vocabulary_size = Column(Integer)
    decoded_tokens = Column(Integer)
    number_shared_tokens = Column(Integer)
    number_unshared_tokens = Column(Integer)
    percentage_subwords = Column(Float)
    percentage_true_words = Column(Float)
    __table_args__ = (UniqueConstraint("tokenizer"),)


###############################################################################
class Vocabulary(Base):
    __tablename__ = "vocabulary"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    tokenizer = Column(String, nullable=False)
    token_id = Column(Integer, nullable=False)
    vocabulary_tokens = Column(String)
    decoded_tokens = Column(String)
    __table_args__ = (UniqueConstraint("tokenizer", "token_id"),)


###############################################################################
class TextDatasetStatistics(Base):
    __tablename__ = "text_dataset_statistics"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    name = Column(String, nullable=False)
    text_id = Column(Integer, ForeignKey("text_dataset.id"), nullable=False)
    words_count = Column(Integer)
    avg_words_length = Column(Float)
    std_words_length = Column(Float)
    __table_args__ = (UniqueConstraint("text_id"),)


###############################################################################
class TextDatasetReports(Base):
    __tablename__ = "text_dataset_reports"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    name = Column(String, nullable=False)
    document_statistics = Column(JSONMapping)
    word_statistics = Column(JSONMapping)
    most_common_words = Column(JSONSequence)
    least_common_words = Column(JSONSequence)
    __table_args__ = (UniqueConstraint("name"),)
