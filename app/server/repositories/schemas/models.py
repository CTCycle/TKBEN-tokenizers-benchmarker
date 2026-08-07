from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Float,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    and_,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from server.repositories.schemas.types import JSONArray, JSONObject, UTCDateTime

###############################################################################
class Base(DeclarativeBase):
    pass

###############################################################################
class Dataset(Base):
    __tablename__ = "dataset"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="loading", server_default="loading")
    document_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    ready_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    __table_args__ = (
        CheckConstraint("length(trim(name)) > 0", name="ck_dataset_name_nonblank"),
        CheckConstraint("status IN ('loading', 'ready')", name="ck_dataset_status"),
        CheckConstraint("document_count >= 0", name="ck_dataset_document_count"),
        CheckConstraint("(status = 'ready' AND ready_at IS NOT NULL) OR (status = 'loading' AND ready_at IS NULL)", name="ck_dataset_ready_timestamp"),
        Index("ix_dataset_status_name", "status", "name"),
    )
    documents: Mapped[list[DatasetDocument]] = relationship(back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True, lazy="raise")
    analysis_sessions: Mapped[list[AnalysisSession]] = relationship(back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True, lazy="raise")
    benchmark_reports: Mapped[list[BenchmarkReport]] = relationship(back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True, lazy="raise")

###############################################################################
class DatasetDocument(Base):
    __tablename__ = "dataset_document"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id", ondelete="CASCADE"), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (
        UniqueConstraint("dataset_id", "ordinal"),
        UniqueConstraint("id", "dataset_id"),
        CheckConstraint("ordinal >= 0", name="ck_dataset_document_ordinal"),
        Index("ix_dataset_document_dataset_id_id", "dataset_id", "id"),
    )
    dataset: Mapped[Dataset] = relationship(back_populates="documents", lazy="raise")
    metric_values: Mapped[list[MetricValue]] = relationship(back_populates="document", passive_deletes=True, lazy="raise", overlaps="metric_values,session")

###############################################################################
class AnalysisSession(Base):
    __tablename__ = "analysis_session"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id", ondelete="CASCADE"), nullable=False)
    session_name: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running", server_default="running")
    report_version: Mapped[int] = mapped_column(Integer, nullable=False, default=2, server_default="2")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONObject(), nullable=False, default=dict, server_default="{}")
    selected_metric_keys: Mapped[list[str]] = mapped_column(JSONArray(), nullable=False, default=list, server_default="[]")
    error_message: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        UniqueConstraint("id", "dataset_id"),
        CheckConstraint("status IN ('running', 'completed', 'failed', 'cancelled')", name="ck_analysis_status"),
        CheckConstraint("report_version > 0", name="ck_analysis_report_version"),
        CheckConstraint("(status = 'running' AND completed_at IS NULL) OR (status <> 'running' AND completed_at IS NOT NULL)", name="ck_analysis_completion_timestamp"),
        Index("ix_analysis_session_dataset_created_id", "dataset_id", "created_at", "id"),
    )
    dataset: Mapped[Dataset] = relationship(back_populates="analysis_sessions", lazy="raise")
    metric_values: Mapped[list[MetricValue]] = relationship(back_populates="session", cascade="all, delete-orphan", passive_deletes=True, lazy="raise", overlaps="metric_values,document")
    histograms: Mapped[list[HistogramArtifact]] = relationship(back_populates="session", cascade="all, delete-orphan", passive_deletes=True, lazy="raise")

###############################################################################
class MetricType(Base):
    __tablename__ = "metric_type"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    scope: Mapped[str] = mapped_column(String(16), nullable=False, default="aggregate", server_default="aggregate")
    value_kind: Mapped[str] = mapped_column(String(16), nullable=False, default="number", server_default="number")
    __table_args__ = (
        CheckConstraint("length(trim(key)) > 0", name="ck_metric_key_nonblank"),
        CheckConstraint("length(trim(category)) > 0", name="ck_metric_category_nonblank"),
        CheckConstraint("length(trim(label)) > 0", name="ck_metric_label_nonblank"),
        CheckConstraint("scope IN ('aggregate', 'per_document')", name="ck_metric_scope"),
        CheckConstraint("value_kind IN ('number', 'text', 'json', 'histogram')", name="ck_metric_value_kind"),
        Index("ix_metric_type_category", "category"),
    )
    metric_values: Mapped[list[MetricValue]] = relationship(back_populates="metric_type", lazy="raise")
    histograms: Mapped[list[HistogramArtifact]] = relationship(back_populates="metric_type", lazy="raise")

###############################################################################
class MetricValue(Base):
    __tablename__ = "metric_value"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_id: Mapped[int] = mapped_column(Integer, nullable=False)
    metric_type_id: Mapped[int] = mapped_column(ForeignKey("metric_type.id"), nullable=False)
    document_id: Mapped[int | None] = mapped_column(Integer)
    numeric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    text_value: Mapped[str | None] = mapped_column(Text)
    json_value: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(JSON(none_as_null=True))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    __table_args__ = (
        ForeignKeyConstraint(["session_id", "dataset_id"], ["analysis_session.id", "analysis_session.dataset_id"], ondelete="CASCADE"),
        ForeignKeyConstraint(["document_id", "dataset_id"], ["dataset_document.id", "dataset_document.dataset_id"], ondelete="CASCADE"),
        CheckConstraint("(CASE WHEN numeric_value IS NOT NULL THEN 1 ELSE 0 END + CASE WHEN text_value IS NOT NULL THEN 1 ELSE 0 END + CASE WHEN json_value IS NOT NULL THEN 1 ELSE 0 END) = 1", name="ck_metric_exactly_one_value"),
        Index("uq_metric_value_aggregate", "session_id", "metric_type_id", unique=True, sqlite_where=and_(document_id.is_(None)), postgresql_where=and_(document_id.is_(None))),
        Index("uq_metric_value_document", "session_id", "metric_type_id", "document_id", unique=True, sqlite_where=and_(document_id.is_not(None)), postgresql_where=and_(document_id.is_not(None))),
        Index("ix_metric_value_session_id", "session_id", "id"),
        Index("ix_metric_value_document_dataset", "document_id", "dataset_id"),
    )
    session: Mapped[AnalysisSession] = relationship(back_populates="metric_values", lazy="raise", overlaps="metric_values,document")
    metric_type: Mapped[MetricType] = relationship(back_populates="metric_values", lazy="raise")
    document: Mapped[DatasetDocument | None] = relationship(back_populates="metric_values", lazy="raise", overlaps="metric_values,session")

###############################################################################
class HistogramArtifact(Base):
    __tablename__ = "histogram_artifact"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("analysis_session.id", ondelete="CASCADE"), nullable=False)
    metric_type_id: Mapped[int] = mapped_column(ForeignKey("metric_type.id"), nullable=False)
    bins: Mapped[list[Any]] = mapped_column(JSONArray(), nullable=False)
    bin_edges: Mapped[list[float]] = mapped_column(JSONArray(), nullable=False)
    counts: Mapped[list[int]] = mapped_column(JSONArray(), nullable=False)
    min_value: Mapped[float] = mapped_column(nullable=False)
    max_value: Mapped[float] = mapped_column(nullable=False)
    mean_value: Mapped[float] = mapped_column(nullable=False)
    median_value: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    __table_args__ = (UniqueConstraint("session_id", "metric_type_id"), CheckConstraint("min_value <= max_value", name="ck_histogram_range"), Index("ix_histogram_artifact_session_id", "session_id", "id"))
    session: Mapped[AnalysisSession] = relationship(back_populates="histograms", lazy="raise")
    metric_type: Mapped[MetricType] = relationship(back_populates="histograms", lazy="raise")

###############################################################################
class Tokenizer(Base):
    __tablename__ = "tokenizer"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    __table_args__ = (CheckConstraint("length(trim(name)) > 0", name="ck_tokenizer_name_nonblank"),)
    reports: Mapped[list[TokenizerReport]] = relationship(back_populates="tokenizer", cascade="all, delete-orphan", passive_deletes=True, lazy="raise")
    vocabularies: Mapped[list[TokenizerVocabulary]] = relationship(back_populates="tokenizer", cascade="all, delete-orphan", passive_deletes=True, lazy="raise")

###############################################################################
class TokenizerVocabulary(Base):
    __tablename__ = "tokenizer_vocabulary"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tokenizer_id: Mapped[int] = mapped_column(ForeignKey("tokenizer.id", ondelete="CASCADE"), nullable=False)
    token_id: Mapped[int] = mapped_column(Integer, nullable=False)
    token: Mapped[str] = mapped_column(Text, nullable=False)
    decoded_token: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (UniqueConstraint("tokenizer_id", "token_id"), CheckConstraint("token_id >= 0", name="ck_tokenizer_token_id"))
    tokenizer: Mapped[Tokenizer] = relationship(back_populates="vocabularies", lazy="raise")

###############################################################################
class TokenizerReport(Base):
    __tablename__ = "tokenizer_report"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tokenizer_id: Mapped[int] = mapped_column(ForeignKey("tokenizer.id", ondelete="CASCADE"), nullable=False, unique=True)
    report_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONObject(), nullable=False)
    token_length_histogram: Mapped[dict[str, Any]] = mapped_column(JSONObject(), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (CheckConstraint("report_version > 0", name="ck_tokenizer_report_version"),)
    tokenizer: Mapped[Tokenizer] = relationship(back_populates="reports", lazy="raise")

###############################################################################
class BenchmarkReport(Base):
    __tablename__ = "benchmark_report"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id", ondelete="CASCADE"), nullable=False)
    report_version: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    methodology_version: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    run_name: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    documents_processed: Mapped[int] = mapped_column(Integer, nullable=False)
    tokenizers_count: Mapped[int] = mapped_column(Integer, nullable=False)
    tokenizers_processed: Mapped[list[str]] = mapped_column(JSONArray(), nullable=False)
    selected_metric_keys: Mapped[list[str]] = mapped_column(JSONArray(), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONObject(), nullable=False)
    __table_args__ = (CheckConstraint("report_version > 0", name="ck_benchmark_report_version"), CheckConstraint("schema_version > 0", name="ck_benchmark_schema_version"), CheckConstraint("documents_processed >= 0 AND tokenizers_count >= 0", name="ck_benchmark_counts"), Index("ix_benchmark_report_dataset_created_id", "dataset_id", "created_at", "id"), Index("ix_benchmark_report_created_id", "created_at", "id"))
    dataset: Mapped[Dataset] = relationship(back_populates="benchmark_reports", lazy="raise")

###############################################################################
class HFAccessKey(Base):
    __tablename__ = "hf_access_keys"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key_value: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")
    __table_args__ = (Index("uq_hf_access_keys_active", "is_active", unique=True, sqlite_where=and_(is_active.is_(True)), postgresql_where=and_(is_active.is_(True))), Index("ix_hf_access_keys_active_id", "is_active", "id"))
