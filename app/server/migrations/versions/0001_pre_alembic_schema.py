"""Record the schema deployed before Alembic was introduced.

This revision is intentionally independent of the live ORM models and remains
part of the historical migration graph. Runtime startup does not infer or
adopt unversioned databases from this revision.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_pre_alembic_schema"
down_revision = None
branch_labels = None
depends_on = None


_CANONICAL_METRIC_CHECK = (
    "(CASE WHEN numeric_value IS NOT NULL THEN 1 ELSE 0 END + "
    "CASE WHEN text_value IS NOT NULL THEN 1 ELSE 0 END + "
    "CASE WHEN json_value IS NOT NULL THEN 1 ELSE 0 END) = 1"
)
_SQLITE_LEGACY_METRIC_CHECK = (
    "(numeric_value IS NOT NULL) + (text_value IS NOT NULL) + "
    "(json_value IS NOT NULL) = 1"
)

###############################################################################
def _is_sqlite() -> bool:
    return op.get_bind().dialect.name == "sqlite"

###############################################################################
def upgrade() -> None:
    active_default = sa.text("'0'")
    metric_value_check = (
        _SQLITE_LEGACY_METRIC_CHECK if _is_sqlite() else _CANONICAL_METRIC_CHECK
    )

    op.create_table(
        "dataset",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default=sa.text("'loading'")),
        sa.Column("document_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("length(trim(name)) > 0", name="ck_dataset_name_nonblank"),
        sa.CheckConstraint("status IN ('loading', 'ready')", name="ck_dataset_status"),
        sa.CheckConstraint("document_count >= 0", name="ck_dataset_document_count"),
        sa.CheckConstraint(
            "(status = 'ready' AND ready_at IS NOT NULL) OR (status = 'loading' AND ready_at IS NULL)",
            name="ck_dataset_ready_timestamp",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_dataset_status_name", "dataset", ["status", "name"])

    op.create_table(
        "dataset_document",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.CheckConstraint("ordinal >= 0", name="ck_dataset_document_ordinal"),
        sa.ForeignKeyConstraint(["dataset_id"], ["dataset.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dataset_id", "ordinal"),
        sa.UniqueConstraint("id", "dataset_id"),
    )
    op.create_index(
        "ix_dataset_document_dataset_id_id",
        "dataset_document",
        ["dataset_id", "id"],
    )

    op.create_table(
        "analysis_session",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("session_name", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default=sa.text("'running'")),
        sa.Column("report_version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("parameters", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("selected_metric_keys", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'cancelled')",
            name="ck_analysis_status",
        ),
        sa.CheckConstraint("report_version > 0", name="ck_analysis_report_version"),
        sa.CheckConstraint(
            "(status = 'running' AND completed_at IS NULL) OR (status <> 'running' AND completed_at IS NOT NULL)",
            name="ck_analysis_completion_timestamp",
        ),
        sa.ForeignKeyConstraint(["dataset_id"], ["dataset.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "dataset_id"),
    )
    op.create_index(
        "ix_analysis_session_dataset_created_id",
        "analysis_session",
        ["dataset_id", "created_at", "id"],
    )

    op.create_table(
        "metric_type",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("scope", sa.String(length=16), nullable=False, server_default=sa.text("'aggregate'")),
        sa.Column("value_kind", sa.String(length=16), nullable=False, server_default=sa.text("'number'")),
        sa.CheckConstraint("length(trim(key)) > 0", name="ck_metric_key_nonblank"),
        sa.CheckConstraint("length(trim(category)) > 0", name="ck_metric_category_nonblank"),
        sa.CheckConstraint("length(trim(label)) > 0", name="ck_metric_label_nonblank"),
        sa.CheckConstraint("scope IN ('aggregate', 'per_document')", name="ck_metric_scope"),
        sa.CheckConstraint(
            "value_kind IN ('number', 'text', 'json', 'histogram')",
            name="ck_metric_value_kind",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )
    op.create_index("ix_metric_type_category", "metric_type", ["category"])

    op.create_table(
        "metric_value",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("metric_type_id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=True),
        sa.Column("numeric_value", sa.Float(), nullable=True),
        sa.Column("text_value", sa.Text(), nullable=True),
        sa.Column("json_value", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(metric_value_check, name="ck_metric_exactly_one_value"),
        sa.ForeignKeyConstraint(
            ["session_id", "dataset_id"],
            ["analysis_session.id", "analysis_session.dataset_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id", "dataset_id"],
            ["dataset_document.id", "dataset_document.dataset_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["metric_type_id"], ["metric_type.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_metric_value_aggregate",
        "metric_value",
        ["session_id", "metric_type_id"],
        unique=True,
        sqlite_where=sa.text("document_id IS NULL"),
        postgresql_where=sa.text("document_id IS NULL"),
    )
    op.create_index(
        "uq_metric_value_document",
        "metric_value",
        ["session_id", "metric_type_id", "document_id"],
        unique=True,
        sqlite_where=sa.text("document_id IS NOT NULL"),
        postgresql_where=sa.text("document_id IS NOT NULL"),
    )
    op.create_index("ix_metric_value_session_id", "metric_value", ["session_id", "id"])
    op.create_index(
        "ix_metric_value_document_dataset",
        "metric_value",
        ["document_id", "dataset_id"],
    )

    op.create_table(
        "histogram_artifact",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("metric_type_id", sa.Integer(), nullable=False),
        sa.Column("bins", sa.JSON(), nullable=False),
        sa.Column("bin_edges", sa.JSON(), nullable=False),
        sa.Column("counts", sa.JSON(), nullable=False),
        sa.Column("min_value", sa.Float(), nullable=False),
        sa.Column("max_value", sa.Float(), nullable=False),
        sa.Column("mean_value", sa.Float(), nullable=False),
        sa.Column("median_value", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("min_value <= max_value", name="ck_histogram_range"),
        sa.ForeignKeyConstraint(["session_id"], ["analysis_session.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["metric_type_id"], ["metric_type.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "metric_type_id"),
    )
    op.create_index(
        "ix_histogram_artifact_session_id",
        "histogram_artifact",
        ["session_id", "id"],
    )

    op.create_table(
        "tokenizer",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("length(trim(name)) > 0", name="ck_tokenizer_name_nonblank"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "tokenizer_vocabulary",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tokenizer_id", sa.Integer(), nullable=False),
        sa.Column("token_id", sa.Integer(), nullable=False),
        sa.Column("token", sa.Text(), nullable=False),
        sa.Column("decoded_token", sa.Text(), nullable=True),
        sa.CheckConstraint("token_id >= 0", name="ck_tokenizer_token_id"),
        sa.ForeignKeyConstraint(["tokenizer_id"], ["tokenizer.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tokenizer_id", "token_id"),
    )

    op.create_table(
        "tokenizer_report",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tokenizer_id", sa.Integer(), nullable=False),
        sa.Column("report_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("token_length_histogram", sa.JSON(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.CheckConstraint("report_version > 0", name="ck_tokenizer_report_version"),
        sa.ForeignKeyConstraint(["tokenizer_id"], ["tokenizer.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tokenizer_id"),
    )

    op.create_table(
        "benchmark_report",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("report_version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("methodology_version", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("run_name", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("documents_processed", sa.Integer(), nullable=False),
        sa.Column("tokenizers_count", sa.Integer(), nullable=False),
        sa.Column("tokenizers_processed", sa.JSON(), nullable=False),
        sa.Column("selected_metric_keys", sa.JSON(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.CheckConstraint("report_version > 0", name="ck_benchmark_report_version"),
        sa.CheckConstraint("schema_version > 0", name="ck_benchmark_schema_version"),
        sa.CheckConstraint(
            "documents_processed >= 0 AND tokenizers_count >= 0",
            name="ck_benchmark_counts",
        ),
        sa.ForeignKeyConstraint(["dataset_id"], ["dataset.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_benchmark_report_dataset_created_id",
        "benchmark_report",
        ["dataset_id", "created_at", "id"],
    )
    op.create_index(
        "ix_benchmark_report_created_id",
        "benchmark_report",
        ["created_at", "id"],
    )

    op.create_table(
        "hf_access_keys",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key_value", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=active_default),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_value"),
    )
    op.create_index(
        "uq_hf_access_keys_active",
        "hf_access_keys",
        ["is_active"],
        unique=True,
        sqlite_where=sa.text("is_active IS 1"),
        postgresql_where=sa.text("is_active IS TRUE"),
    )
    op.create_index("ix_hf_access_keys_active_id", "hf_access_keys", ["is_active", "id"])

###############################################################################
def downgrade() -> None:
    op.drop_index("ix_hf_access_keys_active_id", table_name="hf_access_keys")
    op.drop_index("uq_hf_access_keys_active", table_name="hf_access_keys")
    op.drop_table("hf_access_keys")
    op.drop_index("ix_benchmark_report_created_id", table_name="benchmark_report")
    op.drop_index("ix_benchmark_report_dataset_created_id", table_name="benchmark_report")
    op.drop_table("benchmark_report")
    op.drop_table("tokenizer_report")
    op.drop_table("tokenizer_vocabulary")
    op.drop_table("tokenizer")
    op.drop_index("ix_histogram_artifact_session_id", table_name="histogram_artifact")
    op.drop_table("histogram_artifact")
    op.drop_index("ix_metric_value_document_dataset", table_name="metric_value")
    op.drop_index("ix_metric_value_session_id", table_name="metric_value")
    op.drop_index("uq_metric_value_document", table_name="metric_value")
    op.drop_index("uq_metric_value_aggregate", table_name="metric_value")
    op.drop_table("metric_value")
    op.drop_index("ix_metric_type_category", table_name="metric_type")
    op.drop_table("metric_type")
    op.drop_index("ix_analysis_session_dataset_created_id", table_name="analysis_session")
    op.drop_table("analysis_session")
    op.drop_index("ix_dataset_document_dataset_id_id", table_name="dataset_document")
    op.drop_table("dataset_document")
    op.drop_index("ix_dataset_status_name", table_name="dataset")
    op.drop_table("dataset")
