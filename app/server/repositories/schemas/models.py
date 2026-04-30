from __future__ import annotations

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, mapped_column, relationship

from TKBEN.server.repositories.schemas.types import JSONMapping, JSONSequence

Base = declarative_base()


###############################################################################
class Dataset(Base):
    __tablename__ = "dataset"
    id = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    name = mapped_column(String, nullable=False, unique=True)
    documents = relationship(
        "DatasetDocument", back_populates="dataset", cascade="all, delete-orphan"
    )
    analysis_sessions = relationship(
        "AnalysisSession", back_populates="dataset", cascade="all, delete-orphan"
    )
    benchmark_reports = relationship(
        "BenchmarkReport", back_populates="dataset", cascade="all, delete-orphan"
    )
    validation_reports = relationship(
        "DatasetValidationReport", back_populates="dataset", cascade="all, delete-orphan"
    )


###############################################################################
class Tokenizer(Base):
    __tablename__ = "tokenizer"
    id = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    name = mapped_column(String, nullable=False, unique=True)
    reports = relationship(
        "TokenizerReport", back_populates="tokenizer", cascade="all, delete-orphan"
    )
    vocabularies = relationship(
        "TokenizerVocabulary", back_populates="tokenizer", cascade="all, delete-orphan"
    )


###############################################################################
class HFAccessKey(Base):
    __tablename__ = "hf_access_keys"
    id = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    key_value = mapped_column(String, nullable=False, unique=True)
    created_at = mapped_column(DateTime, nullable=False)
    is_active = mapped_column(Boolean, nullable=False, default=False)
    __table_args__ = (Index("ix_hf_access_keys_is_active", "is_active"),)


###############################################################################
class DatasetDocument(Base):
    __tablename__ = "dataset_document"
    id = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    dataset_id = mapped_column(
        Integer,
        ForeignKey("dataset.id", ondelete="CASCADE"),
        nullable=False,
    )
    text = mapped_column(String, nullable=False)
    __table_args__ = (Index("ix_dataset_document_dataset_id_id", "dataset_id", "id"),)
    dataset = relationship("Dataset", back_populates="documents")
    metric_values = relationship("MetricValue", back_populates="document")


###############################################################################
class AnalysisSession(Base):
    __tablename__ = "analysis_session"
    id = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    dataset_id = mapped_column(
        Integer,
        ForeignKey("dataset.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_name = mapped_column(String, nullable=True)
    status = mapped_column(String, nullable=False, default="completed")
    report_version = mapped_column(Integer, nullable=False, default=2)
    created_at = mapped_column(DateTime, nullable=False)
    completed_at = mapped_column(DateTime, nullable=True)
    parameters = mapped_column(JSONMapping)
    selected_metric_keys = mapped_column(JSONSequence)
    __table_args__ = (
        Index("ix_analysis_session_dataset_id", "dataset_id"),
        Index(
            "ix_analysis_session_dataset_id_created_at",
            "dataset_id",
            "created_at",
        ),
    )
    dataset = relationship("Dataset", back_populates="analysis_sessions")
    metric_values = relationship(
        "MetricValue", back_populates="session", cascade="all, delete-orphan"
    )
    histograms = relationship(
        "HistogramArtifact", back_populates="session", cascade="all, delete-orphan"
    )


###############################################################################
class MetricType(Base):
    __tablename__ = "metric_type"
    id = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    key = mapped_column(String, nullable=False, unique=True)
    category = mapped_column(String, nullable=False)
    label = mapped_column(String, nullable=False)
    description = mapped_column(String, nullable=True)
    scope = mapped_column(String, nullable=False, default="aggregate")
    value_kind = mapped_column(String, nullable=False, default="number")
    __table_args__ = (Index("ix_metric_type_category", "category"),)
    metric_values = relationship("MetricValue", back_populates="metric_type")
    histograms = relationship("HistogramArtifact", back_populates="metric_type")


###############################################################################
class MetricValue(Base):
    __tablename__ = "metric_value"
    id = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    session_id = mapped_column(
        Integer,
        ForeignKey("analysis_session.id", ondelete="CASCADE"),
        nullable=False,
    )
    metric_type_id = mapped_column(
        Integer,
        ForeignKey("metric_type.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id = mapped_column(
        Integer,
        ForeignKey("dataset_document.id", ondelete="CASCADE"),
        nullable=True,
    )
    numeric_value = mapped_column(Float, nullable=True)
    text_value = mapped_column(String, nullable=True)
    json_value = mapped_column(JSONMapping, nullable=True)
    __table_args__ = (
        UniqueConstraint("session_id", "metric_type_id", "document_id"),
        Index("ix_metric_value_session_metric", "session_id", "metric_type_id"),
        Index("ix_metric_value_document", "document_id"),
    )
    session = relationship("AnalysisSession", back_populates="metric_values")
    metric_type = relationship("MetricType", back_populates="metric_values")
    document = relationship("DatasetDocument", back_populates="metric_values")


###############################################################################
class HistogramArtifact(Base):
    __tablename__ = "histogram_artifact"
    id = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    session_id = mapped_column(
        Integer,
        ForeignKey("analysis_session.id", ondelete="CASCADE"),
        nullable=False,
    )
    metric_type_id = mapped_column(
        Integer,
        ForeignKey("metric_type.id", ondelete="CASCADE"),
        nullable=False,
    )
    bins = mapped_column(JSONSequence)
    bin_edges = mapped_column(JSONSequence)
    counts = mapped_column(JSONSequence)
    min_value = mapped_column(Float, nullable=False, default=0.0)
    max_value = mapped_column(Float, nullable=False, default=0.0)
    mean_value = mapped_column(Float, nullable=False, default=0.0)
    median_value = mapped_column(Float, nullable=False, default=0.0)
    __table_args__ = (
        UniqueConstraint("session_id", "metric_type_id"),
        Index("ix_histogram_artifact_session", "session_id"),
    )
    session = relationship("AnalysisSession", back_populates="histograms")
    metric_type = relationship("MetricType", back_populates="histograms")


class TokenizerVocabulary(Base):
    __tablename__ = "tokenizer_vocabulary"
    id = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    tokenizer_id = mapped_column(
        Integer,
        ForeignKey("tokenizer.id", ondelete="CASCADE"),
        nullable=False,
    )
    token_id = mapped_column(Integer, nullable=False)
    vocabulary_tokens = mapped_column(String)
    decoded_tokens = mapped_column(String)
    __table_args__ = (
        UniqueConstraint("tokenizer_id", "token_id"),
        Index("ix_tokenizer_vocabulary_tokenizer_id", "tokenizer_id"),
    )
    tokenizer = relationship("Tokenizer", back_populates="vocabularies")


###############################################################################
class TokenizerReport(Base):
    __tablename__ = "tokenizer_report"
    id = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    tokenizer_id = mapped_column(
        Integer,
        ForeignKey("tokenizer.id", ondelete="CASCADE"),
        nullable=False,
    )
    report_version = mapped_column(Integer, nullable=False, default=1)
    created_at = mapped_column(DateTime, nullable=False)
    metadata_json = mapped_column("metadata", JSONMapping)
    token_length_histogram = mapped_column(JSONMapping)
    description = mapped_column(String)
    __table_args__ = (
        Index("ix_tokenizer_report_tokenizer_id", "tokenizer_id"),
        Index(
            "ix_tokenizer_report_tokenizer_id_created_at", "tokenizer_id", "created_at"
        ),
    )
    tokenizer = relationship("Tokenizer", back_populates="reports")


###############################################################################
class BenchmarkReport(Base):
    __tablename__ = "benchmark_report"
    id = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    dataset_id = mapped_column(
        Integer,
        ForeignKey("dataset.id", ondelete="CASCADE"),
        nullable=False,
    )
    report_version = mapped_column(Integer, nullable=False, default=1)
    created_at = mapped_column(DateTime, nullable=False)
    run_name = mapped_column(String, nullable=True)
    selected_metric_keys = mapped_column(JSONSequence)
    payload = mapped_column(JSONMapping)
    __table_args__ = (
        Index("ix_benchmark_report_dataset_id", "dataset_id"),
        Index("ix_benchmark_report_created_at", "created_at"),
    )
    dataset = relationship("Dataset", back_populates="benchmark_reports")


class DatasetValidationReport(Base):
    __tablename__ = "dataset_validation_report"
    id = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    dataset_id = mapped_column(
        Integer,
        ForeignKey("dataset.id", ondelete="CASCADE"),
        nullable=False,
    )
    report_version = mapped_column(Integer, nullable=False, default=1)
    created_at = mapped_column(DateTime, nullable=False)
    aggregate_statistics = mapped_column(JSONMapping)
    document_histogram = mapped_column(JSONMapping)
    word_histogram = mapped_column(JSONMapping)
    most_common_words = mapped_column(JSONSequence)
    least_common_words = mapped_column(JSONSequence)
    longest_words = mapped_column(JSONSequence)
    shortest_words = mapped_column(JSONSequence)
    word_cloud_terms = mapped_column(JSONSequence)
    per_document_stats = mapped_column(JSONMapping)
    __table_args__ = (
        Index("ix_dataset_validation_report_dataset_id", "dataset_id"),
        Index(
            "ix_dataset_validation_report_dataset_id_id_desc",
            "dataset_id",
            "id",
        ),
    )
    dataset = relationship("Dataset", back_populates="validation_reports")

