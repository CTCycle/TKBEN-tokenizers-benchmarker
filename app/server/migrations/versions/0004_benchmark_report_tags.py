"""Add mutable tags to benchmark reports."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0004_benchmark_report_tags"
down_revision = "0003_canonical_state_cleanup"
branch_labels = None
depends_on = None


###############################################################################
def upgrade() -> None:
    op.add_column(
        "benchmark_report",
        sa.Column(
            "tags",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )


###############################################################################
def downgrade() -> None:
    op.drop_column("benchmark_report", "tags")
