"""add_order_source_and_menu_item_active

Revision ID: 7a03688b7bbc
Revises: 802abd40fb6d
Create Date: 2026-10-24 12:43:47.361718

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7a03688b7bbc'
down_revision: Union[str, Sequence[str], None] = '802abd40fb6d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Menu items can be retired without deleting them (history is kept).
    op.add_column('menu_items', sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False))
    # One active dish per name (ignoring case) per restaurant; fails if the
    # menu already has active duplicates, which must be resolved first.
    op.create_index('uq_menu_items_restaurant_active_name', 'menu_items',
                    ['restaurant_id', sa.text('lower(name)')], unique=True,
                    postgresql_where=sa.text('is_active'))
    # Order channel.  Orders created before this revision get 'UNKNOWN' (their
    # channel was never recorded); new rows default to 'API'.
    op.add_column('orders', sa.Column('source', sa.String(length=20), nullable=True))
    op.execute("UPDATE orders SET source = 'UNKNOWN'")
    op.alter_column('orders', 'source', existing_type=sa.String(length=20),
                    nullable=False, server_default='API')
    op.create_check_constraint(op.f('ck_orders_source_valid'), 'orders',
                               "source IN ('AI_VOICE', 'AI_CHAT', 'WALK_IN_POS', 'API', 'UNKNOWN')")
    # Walk-in sales may have no contact number.
    op.alter_column('orders', 'customer_phone',
               existing_type=sa.VARCHAR(length=20),
               nullable=True)


def downgrade() -> None:
    """Downgrade schema.

    Refuses (rather than inventing phone numbers) if any order has no
    contact number; resolve those rows first.
    """
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM orders WHERE customer_phone IS NULL) THEN "
        "RAISE EXCEPTION 'orders without customer_phone exist; cannot restore NOT NULL'; "
        "END IF; END $$"
    )
    op.alter_column('orders', 'customer_phone',
               existing_type=sa.VARCHAR(length=20),
               nullable=False)
    op.drop_constraint(op.f('ck_orders_source_valid'), 'orders', type_='check')
    op.drop_column('orders', 'source')
    op.drop_index('uq_menu_items_restaurant_active_name', table_name='menu_items',
                  postgresql_where=sa.text('is_active'))
    op.drop_column('menu_items', 'is_active')
