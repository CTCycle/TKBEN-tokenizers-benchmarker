from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

from TKBEN.server.repositories.schemas.types import JSONMapping, JSONSequence

Base = declarative_base()


###############################################################################
class Dataset(Base):
    __tablename__ = "dataset"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    name = Column(String, nullable=False, unique=True)


###############################################################################
class Tokenizer(Base):
    __tablename__ = "tokenizer"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    name = Column(String, nullable=False, unique=True)


###############################################################################
class HFAccessKey(Base):
    __tablename__ = "hf_access_keys"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    key_value = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, nullable=False, default=False)
    __table_args__ = (
        Index("ix_hf_access_keys_is_active", "is_active"),
    )


###############################################################################
class DatasetDocument(Base):
    __tablename__ = "dataset_document"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    dataset_id = Column(
        Integer,
        ForeignKey("dataset.id", ondelete="CASCADE"),
        nullable=False,
    )
    text = Column(String, nullable=False)
    __table_args__ = (
        Index("ix_dataset_document_dataset_id_id", "dataset_id", "id"),
    )


###############################################################################
class DatasetDocumentStatistics(Base):
    __tablename__ = "dataset_document_statistics"
    document_id = Column(
        Integer,
        ForeignKey("dataset_document.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    words_count = Column(Integer)
    avg_words_length = Column(Float)
    std_words_length = Column(Float)


###############################################################################
class DatasetReport(Base):
    __tablename__ = "dataset_report"
    dataset_id = Column(
        Integer,
        ForeignKey("dataset.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    document_statistics = Column(JSONMapping)
    word_statistics = Column(JSONMapping)
    most_common_words = Column(JSONSequence)
    least_common_words = Column(JSONSequence)


###############################################################################
class DatasetValidationReport(Base):
    __tablename__ = "dataset_validation_report"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    dataset_id = Column(
        Integer,
        ForeignKey("dataset.id", ondelete="CASCADE"),
        nullable=False,
    )
    report_version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False)
    aggregate_statistics = Column(JSONMapping)
    document_histogram = Column(JSONMapping)
    word_histogram = Column(JSONMapping)
    most_common_words = Column(JSONSequence)
    least_common_words = Column(JSONSequence)
    longest_words = Column(JSONSequence)
    shortest_words = Column(JSONSequence)
    word_cloud_terms = Column(JSONSequence)
    per_document_stats = Column(JSONMapping)
    __table_args__ = (
        Index("ix_dataset_validation_report_dataset_id", "dataset_id"),
        Index(
            "ix_dataset_validation_report_dataset_id_created_at",
            "dataset_id",
            "created_at",
        ),
    )


###############################################################################
class AnalysisSession(Base):
    __tablename__ = "analysis_session"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    dataset_id = Column(
        Integer,
        ForeignKey("dataset.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_name = Column(String, nullable=True)
    status = Column(String, nullable=False, default="completed")
    report_version = Column(Integer, nullable=False, default=2)
    created_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    parameters = Column(JSONMapping)
    selected_metric_keys = Column(JSONSequence)
    __table_args__ = (
        Index("ix_analysis_session_dataset_id", "dataset_id"),
        Index(
            "ix_analysis_session_dataset_id_created_at",
            "dataset_id",
            "created_at",
        ),
    )


###############################################################################
class MetricType(Base):
    __tablename__ = "metric_type"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    key = Column(String, nullable=False, unique=True)
    category = Column(String, nullable=False)
    label = Column(String, nullable=False)
    description = Column(String, nullable=True)
    scope = Column(String, nullable=False, default="aggregate")
    value_kind = Column(String, nullable=False, default="number")
    __table_args__ = (
        Index("ix_metric_type_category", "category"),
    )


###############################################################################
class MetricValue(Base):
    __tablename__ = "metric_value"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    session_id = Column(
        Integer,
        ForeignKey("analysis_session.id", ondelete="CASCADE"),
        nullable=False,
    )
    metric_type_id = Column(
        Integer,
        ForeignKey("metric_type.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id = Column(
        Integer,
        ForeignKey("dataset_document.id", ondelete="CASCADE"),
        nullable=True,
    )
    numeric_value = Column(Float, nullable=True)
    text_value = Column(String, nullable=True)
    json_value = Column(JSONMapping, nullable=True)
    __table_args__ = (
        UniqueConstraint("session_id", "metric_type_id", "document_id"),
        Index("ix_metric_value_session_metric", "session_id", "metric_type_id"),
        Index("ix_metric_value_document", "document_id"),
    )


###############################################################################
class HistogramArtifact(Base):
    __tablename__ = "histogram_artifact"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    session_id = Column(
        Integer,
        ForeignKey("analysis_session.id", ondelete="CASCADE"),
        nullable=False,
    )
    metric_type_id = Column(
        Integer,
        ForeignKey("metric_type.id", ondelete="CASCADE"),
        nullable=False,
    )
    bins = Column(JSONSequence)
    bin_edges = Column(JSONSequence)
    counts = Column(JSONSequence)
    min_value = Column(Float, nullable=False, default=0.0)
    max_value = Column(Float, nullable=False, default=0.0)
    mean_value = Column(Float, nullable=False, default=0.0)
    median_value = Column(Float, nullable=False, default=0.0)
    __table_args__ = (
        UniqueConstraint("session_id", "metric_type_id"),
        Index("ix_histogram_artifact_session", "session_id"),
    )


###############################################################################
class TokenizationDocumentStats(Base):
    __tablename__ = "tokenization_document_stats"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    tokenizer_id = Column(
        Integer,
        ForeignKey("tokenizer.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id = Column(
        Integer,
        ForeignKey("dataset_document.id", ondelete="CASCADE"),
        nullable=False,
    )
    tokens_count = Column(Integer)
    tokens_to_words_ratio = Column(Float)
    bytes_per_token = Column(Float)
    boundary_preservation_rate = Column(Float)
    round_trip_token_fidelity = Column(Float)
    round_trip_text_fidelity = Column(Float)
    determinism_stability = Column(Float)
    bytes_per_character = Column(Float)
    __table_args__ = (
        UniqueConstraint("tokenizer_id", "document_id"),
        Index("ix_tokenization_document_stats_tokenizer_id", "tokenizer_id"),
        Index("ix_tokenization_document_stats_document_id", "document_id"),
    )


###############################################################################
class TokenizationDatasetStats(Base):
    __tablename__ = "tokenization_dataset_stats"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    tokenizer_id = Column(
        Integer,
        ForeignKey("tokenizer.id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_id = Column(
        Integer,
        ForeignKey("dataset.id", ondelete="CASCADE"),
        nullable=False,
    )
    tokenization_speed_tps = Column(Float)
    throughput_chars_per_sec = Column(Float)
    model_size_mb = Column(Float)
    vocabulary_size = Column(Integer)
    subword_fertility = Column(Float)
    oov_rate = Column(Float)
    word_recovery_rate = Column(Float)
    __table_args__ = (
        UniqueConstraint("tokenizer_id", "dataset_id"),
        Index("ix_tokenization_dataset_stats_tokenizer_id", "tokenizer_id"),
        Index("ix_tokenization_dataset_stats_dataset_id", "dataset_id"),
    )


###############################################################################
class TokenizationDatasetStatsDetail(Base):
    __tablename__ = "tokenization_dataset_stats_detail"
    global_stats_id = Column(
        Integer,
        ForeignKey("tokenization_dataset_stats.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
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


###############################################################################
class TokenizerVocabularyStatistics(Base):
    __tablename__ = "tokenizer_vocabulary_statistics"
    tokenizer_id = Column(
        Integer,
        ForeignKey("tokenizer.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    vocabulary_size = Column(Integer)
    decoded_tokens = Column(Integer)
    number_shared_tokens = Column(Integer)
    number_unshared_tokens = Column(Integer)
    percentage_subwords = Column(Float)
    percentage_true_words = Column(Float)


###############################################################################
class TokenizerVocabulary(Base):
    __tablename__ = "tokenizer_vocabulary"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    tokenizer_id = Column(
        Integer,
        ForeignKey("tokenizer.id", ondelete="CASCADE"),
        nullable=False,
    )
    token_id = Column(Integer, nullable=False)
    vocabulary_tokens = Column(String)
    decoded_tokens = Column(String)
    __table_args__ = (
        UniqueConstraint("tokenizer_id", "token_id"),
        Index("ix_tokenizer_vocabulary_tokenizer_id", "tokenizer_id"),
    )


###############################################################################
class TokenizerReport(Base):
    __tablename__ = "tokenizer_report"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    tokenizer_id = Column(
        Integer,
        ForeignKey("tokenizer.id", ondelete="CASCADE"),
        nullable=False,
    )
    report_version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False)
    metadata_json = Column("metadata", JSONMapping)
    token_length_histogram = Column(JSONMapping)
    description = Column(String)
    __table_args__ = (
        Index("ix_tokenizer_report_tokenizer_id", "tokenizer_id"),
        Index("ix_tokenizer_report_tokenizer_id_created_at", "tokenizer_id", "created_at"),
    )
