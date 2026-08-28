"""Normalize the pre-Alembic schema to the current application contract."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_current_schema"
down_revision = "0001_pre_alembic_schema"
branch_labels = None
depends_on = None


_CANONICAL_METRIC_CHECK = (
    "(CASE WHEN numeric_value IS NOT NULL THEN 1 ELSE 0 END + "
    "CASE WHEN text_value IS NOT NULL THEN 1 ELSE 0 END + "
    "CASE WHEN json_value IS NOT NULL THEN 1 ELSE 0 END) = 1"
)
_LEGACY_METRIC_CHECK = (
    "(numeric_value IS NOT NULL) + (text_value IS NOT NULL) + "
    "(json_value IS NOT NULL) = 1"
)

###############################################################################
def _is_sqlite() -> bool:
    return op.get_bind().dialect.name == "sqlite"

###############################################################################
def _alter_metric_value_check(expression: str) -> None:
    if _is_sqlite():
        with op.batch_alter_table("metric_value", recreate="always") as batch:
            batch.drop_constraint("ck_metric_exactly_one_value", type_="check")
            batch.create_check_constraint("ck_metric_exactly_one_value", expression)
        return

    op.drop_constraint("ck_metric_exactly_one_value", "metric_value", type_="check")
    op.create_check_constraint("ck_metric_exactly_one_value", expression, "metric_value")

###############################################################################
def _alter_report_version_default(default: str) -> None:
    server_default = sa.text(f"'{default}'")
    if _is_sqlite():
        with op.batch_alter_table("analysis_session", recreate="always") as batch:
            batch.alter_column(
                "report_version",
                existing_type=sa.Integer(),
                existing_nullable=False,
                existing_server_default=sa.text("1" if default == "2" else "2"),
                server_default=server_default,
            )
        return

    op.alter_column(
        "analysis_session",
        "report_version",
        existing_type=sa.Integer(),
        existing_nullable=False,
        existing_server_default=sa.text("1" if default == "2" else "2"),
        server_default=server_default,
    )

###############################################################################
def upgrade() -> None:
    _alter_metric_value_check(_CANONICAL_METRIC_CHECK)
    _alter_report_version_default("2")

###############################################################################
def downgrade() -> None:
    _alter_report_version_default("1")
    _alter_metric_value_check(
        _LEGACY_METRIC_CHECK if _is_sqlite() else _CANONICAL_METRIC_CHECK
    )
