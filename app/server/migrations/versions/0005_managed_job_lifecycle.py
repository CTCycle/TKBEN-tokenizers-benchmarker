"""Persist managed job lifecycle state."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0005_managed_job_lifecycle"
down_revision = "0004_benchmark_report_tags"
branch_labels = None
depends_on = None

###############################################################################
def upgrade() -> None:
    op.create_table(
        "managed_job",
        sa.Column("job_id", sa.String(length=8), nullable=False),
        sa.Column("job_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("failure_reason", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_managed_job_status",
        ),
        sa.CheckConstraint(
            "progress >= 0 AND progress <= 100", name="ck_managed_job_progress"
        ),
        sa.CheckConstraint(
            "(status IN ('pending', 'running') AND completed_at IS NULL) OR "
            "(status IN ('completed', 'failed', 'cancelled') AND completed_at IS NOT NULL)",
            name="ck_managed_job_completion_timestamp",
        ),
        sa.PrimaryKeyConstraint("job_id"),
    )
    op.create_index(
        "ix_managed_job_status_created_at",
        "managed_job",
        ["status", "created_at"],
    )
    op.create_index("ix_managed_job_completed_at", "managed_job", ["completed_at"])

###############################################################################
def downgrade() -> None:
    op.drop_index("ix_managed_job_completed_at", table_name="managed_job")
    op.drop_index("ix_managed_job_status_created_at", table_name="managed_job")
    op.drop_table("managed_job")
