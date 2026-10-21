# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

"""Persist dashboard UI preferences in the active database."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0007_dashboard_ui_preferences"
down_revision = "0006_database_runtime_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("application_runtime_settings") as batch_op:
        batch_op.add_column(
            sa.Column(
                "dashboard_revision",
                sa.Integer(),
                server_default="0",
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column(
                "dashboard_preferences",
                sa.JSON(),
                server_default="{}",
                nullable=False,
            )
        )
        batch_op.create_check_constraint(
            "ck_dashboard_settings_revision", "dashboard_revision >= 0"
        )


def downgrade() -> None:
    with op.batch_alter_table("application_runtime_settings") as batch_op:
        batch_op.drop_constraint(
            "ck_dashboard_settings_revision", type_="check"
        )
        batch_op.drop_column("dashboard_preferences")
        batch_op.drop_column("dashboard_revision")
