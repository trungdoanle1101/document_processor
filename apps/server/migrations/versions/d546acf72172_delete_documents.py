"""delete documents

Revision ID: d546acf72172
Revises: 84861029d433
Create Date: 2026-11-11 09:03:02.109721

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d546acf72172"
down_revision: str | Sequence[str] | None = "84861029d433"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Allow the 'deleting' document status and the 'delete' job kind."""
    op.drop_constraint("document_status", "documents", type_="check")
    op.create_check_constraint(
        "document_status", "documents", "status IN ('draft', 'active', 'deleting')"
    )
    op.drop_constraint("job_kind", "indexing_jobs", type_="check")
    op.create_check_constraint(
        "job_kind",
        "indexing_jobs",
        "kind IN ('prepare', 'index', 'sync_attributes', 'delete')",
    )


def downgrade() -> None:
    """Keep the wider constraints: removing a value would fail or lose rows that use it."""
    # Code from before this revision never writes 'deleting' or 'delete', so the
    # wider CHECK can't reject anything that code does
