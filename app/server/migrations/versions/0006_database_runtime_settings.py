# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

"""Persist runtime application settings in the active database."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0006_database_runtime_settings"
down_revision = "0005_managed_job_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "application_runtime_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("overrides", sa.JSON(), server_default="{}", nullable=False),
        sa.CheckConstraint("id = 1", name="ck_runtime_settings_singleton"),
        sa.CheckConstraint(
            "schema_version > 0", name="ck_runtime_settings_schema_version"
        ),
        sa.CheckConstraint("revision >= 0", name="ck_runtime_settings_revision"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("application_runtime_settings")
