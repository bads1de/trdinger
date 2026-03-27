"""Add experiment_id column to ga_experiments

Revision ID: a1b2c3d4e5f6
Revises: 515fd9c63048
Create Date: 2026-03-28 06:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '515fd9c63048'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'ga_experiments',
        sa.Column('experiment_id', sa.String(length=255), nullable=True),
    )
    op.create_unique_constraint(
        'uq_ga_experiments_experiment_id',
        'ga_experiments',
        ['experiment_id'],
    )
    op.create_index(
        'idx_ga_experiments_experiment_id',
        'ga_experiments',
        ['experiment_id'],
    )


def downgrade() -> None:
    op.drop_index('idx_ga_experiments_experiment_id', table_name='ga_experiments')
    op.drop_constraint('uq_ga_experiments_experiment_id', 'ga_experiments', type_='unique')
    op.drop_column('ga_experiments', 'experiment_id')
