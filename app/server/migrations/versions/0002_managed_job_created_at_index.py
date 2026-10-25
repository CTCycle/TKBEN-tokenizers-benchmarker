# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

"""Index managed jobs by creation time.

Job history is capped by count, so listing and pruning order jobs by
`created_at`.
"""

from __future__ import annotations

from alembic import op


revision = "0002_managed_job_created_at_index"
down_revision = "0001_current_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("managed_job", schema=None) as batch_op:
        batch_op.create_index("ix_managed_job_created_at", ["created_at"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("managed_job", schema=None) as batch_op:
        batch_op.drop_index("ix_managed_job_created_at")
