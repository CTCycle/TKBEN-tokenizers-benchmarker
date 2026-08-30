"""Make canonical report state and metric identities explicit."""

from __future__ import annotations

import json

from alembic import op
import sqlalchemy as sa


revision = "0003_canonical_state_cleanup"
down_revision = "0002_current_schema"
branch_labels = None
depends_on = None

DATASET_REPORT_VERSION = 2
TOKENIZER_REPORT_VERSION = 1
BENCHMARK_REPORT_VERSION = 5
BENCHMARK_SCHEMA_VERSION = 3


###############################################################################
def _purge_incompatible_reports() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "DELETE FROM metric_value WHERE session_id IN "
            "(SELECT id FROM analysis_session WHERE report_version != :version)"
        ),
        {"version": DATASET_REPORT_VERSION},
    )
    bind.execute(
        sa.text(
            "DELETE FROM histogram_artifact WHERE session_id IN "
            "(SELECT id FROM analysis_session WHERE report_version != :version)"
        ),
        {"version": DATASET_REPORT_VERSION},
    )
    bind.execute(
        sa.text("DELETE FROM analysis_session WHERE report_version != :version"),
        {"version": DATASET_REPORT_VERSION},
    )
    bind.execute(
        sa.text(
            "DELETE FROM benchmark_report WHERE report_version != :report_version "
            "OR schema_version != :schema_version"
        ),
        {
            "report_version": BENCHMARK_REPORT_VERSION,
            "schema_version": BENCHMARK_SCHEMA_VERSION,
        },
    )
    bind.execute(
        sa.text("DELETE FROM tokenizer_report WHERE report_version != :version"),
        {"version": TOKENIZER_REPORT_VERSION},
    )


###############################################################################
def _normalize_benchmark_payloads() -> None:
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT id, payload FROM benchmark_report WHERE report_version = :report_version "
            "AND schema_version = :schema_version"
        ),
        {
            "report_version": BENCHMARK_REPORT_VERSION,
            "schema_version": BENCHMARK_SCHEMA_VERSION,
        },
    ).all()
    benchmark_report = sa.table(
        "benchmark_report",
        sa.column("id", sa.Integer),
        sa.column("payload", sa.JSON),
    )
    summary_fields = {
        "report_id",
        "report_version",
        "schema_version",
        "methodology_version",
        "created_at",
        "run_name",
        "status",
        "documents_processed",
        "tokenizers_count",
        "tokenizers_processed",
        "selected_metric_keys",
        "dataset_name",
    }
    for report_id, payload in rows:
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Benchmark report {report_id} has a non-object JSON payload."
            )
        details = {
            key: value for key, value in payload.items() if key not in summary_fields
        }
        bind.execute(
            sa.update(benchmark_report)
            .where(benchmark_report.c.id == report_id)
            .values(payload=details)
        )


###############################################################################
def _migrate_metric_values() -> None:
    bind = op.get_bind()
    op.add_column("metric_value", sa.Column("metric_key", sa.String(255)))
    bind.execute(
        sa.text(
            "UPDATE metric_value SET metric_key = "
            "(SELECT key FROM metric_type WHERE metric_type.id = metric_value.metric_type_id)"
        )
    )
    op.drop_index("uq_metric_value_aggregate", table_name="metric_value")
    op.drop_index("uq_metric_value_document", table_name="metric_value")
    with op.batch_alter_table("metric_value", recreate="always") as batch:
        batch.drop_column("metric_type_id")
        batch.alter_column(
            "metric_key",
            existing_type=sa.String(255),
            nullable=False,
        )
        batch.create_check_constraint(
            "ck_metric_value_key_nonblank",
            "length(trim(metric_key)) > 0",
        )
    op.create_index(
        "uq_metric_value_aggregate",
        "metric_value",
        ["session_id", "metric_key"],
        unique=True,
        sqlite_where=sa.text("document_id IS NULL"),
        postgresql_where=sa.text("document_id IS NULL"),
    )
    op.create_index(
        "uq_metric_value_document",
        "metric_value",
        ["session_id", "metric_key", "document_id"],
        unique=True,
        sqlite_where=sa.text("document_id IS NOT NULL"),
        postgresql_where=sa.text("document_id IS NOT NULL"),
    )


###############################################################################
def _migrate_histograms() -> None:
    bind = op.get_bind()
    op.add_column("histogram_artifact", sa.Column("metric_key", sa.String(255)))
    bind.execute(
        sa.text(
            "UPDATE histogram_artifact SET metric_key = "
            "(SELECT key FROM metric_type WHERE metric_type.id = histogram_artifact.metric_type_id)"
        )
    )
    with op.batch_alter_table("histogram_artifact", recreate="always") as batch:
        batch.drop_column("metric_type_id")
        batch.alter_column(
            "metric_key",
            existing_type=sa.String(255),
            nullable=False,
        )
        batch.create_unique_constraint(
            "uq_histogram_artifact_session_metric_key",
            ["session_id", "metric_key"],
        )
        batch.create_check_constraint(
            "ck_histogram_metric_key_nonblank",
            "length(trim(metric_key)) > 0",
        )


###############################################################################
def _remove_historical_custom_tokenizers() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "DELETE FROM tokenizer_vocabulary WHERE tokenizer_id IN "
            "(SELECT id FROM tokenizer WHERE name LIKE 'CUSTOM_%')"
        )
    )
    bind.execute(
        sa.text(
            "DELETE FROM tokenizer_report WHERE tokenizer_id IN "
            "(SELECT id FROM tokenizer WHERE name LIKE 'CUSTOM_%')"
        )
    )
    bind.execute(sa.text("DELETE FROM tokenizer WHERE name LIKE 'CUSTOM_%'"))


###############################################################################
def _remove_metric_catalog() -> None:
    op.drop_index("ix_metric_type_category", table_name="metric_type")
    op.drop_table("metric_type")


###############################################################################
def upgrade() -> None:
    _purge_incompatible_reports()
    _normalize_benchmark_payloads()
    _migrate_metric_values()
    _migrate_histograms()
    _remove_metric_catalog()
    _remove_historical_custom_tokenizers()
    with op.batch_alter_table("tokenizer", recreate="always") as batch:
        batch.add_column(
            sa.Column(
                "source",
                sa.String(32),
                nullable=False,
                server_default=sa.text("'huggingface'"),
            )
        )
        batch.create_check_constraint(
            "ck_tokenizer_source",
            "source IN ('huggingface', 'custom')",
        )


###############################################################################
def downgrade() -> None:
    raise NotImplementedError(
        "The canonical state cleanup is intentionally irreversible."
    )
