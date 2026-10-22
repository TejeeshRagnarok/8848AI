"""create_orders_and_order_items

Revision ID: 09b7b2a79c59
Revises: 25620f605c18
Create Date: 2026-10-22 12:15:40.959927

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '09b7b2a79c59'
down_revision: Union[str, Sequence[str], None] = '25620f605c18'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Order numbers ("ORD-100001") come from this sequence via the
    # orders.order_number server default, so it must exist first.
    op.execute(sa.schema.CreateSequence(sa.Sequence('order_number_seq', start=100001)))
    op.create_table('orders',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('restaurant_id', sa.Uuid(), nullable=False),
    sa.Column('order_number', sa.String(length=20), server_default=sa.text("'ORD-' || nextval('order_number_seq')"), nullable=False),
    sa.Column('customer_name', sa.String(length=100), nullable=True),
    sa.Column('customer_phone', sa.String(length=20), nullable=False),
    sa.Column('status', sa.String(length=20), server_default='pending', nullable=False),
    sa.Column('subtotal_paise', sa.BigInteger(), nullable=False),
    sa.Column('total_paise', sa.BigInteger(), nullable=False),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('idempotency_key', sa.String(length=100), nullable=False),
    sa.Column('request_hash', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('statement_timestamp()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('statement_timestamp()'), nullable=False),
    sa.CheckConstraint("status IN ('pending', 'confirmed', 'preparing', 'ready', 'completed', 'cancelled')", name=op.f('ck_orders_status_valid')),
    sa.CheckConstraint('subtotal_paise >= 0', name=op.f('ck_orders_subtotal_paise_nonnegative')),
    sa.CheckConstraint('total_paise >= 0', name=op.f('ck_orders_total_paise_nonnegative')),
    sa.ForeignKeyConstraint(['restaurant_id'], ['restaurants.id'], name=op.f('fk_orders_restaurant_id_restaurants'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_orders')),
    sa.UniqueConstraint('order_number', name=op.f('uq_orders_order_number')),
    sa.UniqueConstraint('restaurant_id', 'idempotency_key', name='uq_orders_restaurant_id_idempotency_key')
    )
    op.create_index('ix_orders_restaurant_id_created_at', 'orders', ['restaurant_id', 'created_at'], unique=False)
    op.create_table('order_items',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('order_id', sa.Uuid(), nullable=False),
    sa.Column('menu_item_id', sa.Uuid(), nullable=True),
    sa.Column('item_name', sa.String(length=200), nullable=False),
    sa.Column('unit_price_paise', sa.Integer(), nullable=False),
    sa.Column('quantity', sa.Integer(), nullable=False),
    sa.Column('line_total_paise', sa.BigInteger(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('statement_timestamp()'), nullable=False),
    sa.CheckConstraint('line_total_paise = unit_price_paise * quantity', name=op.f('ck_order_items_line_total_matches')),
    sa.CheckConstraint('quantity > 0', name=op.f('ck_order_items_quantity_positive')),
    sa.CheckConstraint('unit_price_paise > 0', name=op.f('ck_order_items_unit_price_paise_positive')),
    sa.ForeignKeyConstraint(['menu_item_id'], ['menu_items.id'], name=op.f('fk_order_items_menu_item_id_menu_items'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['order_id'], ['orders.id'], name=op.f('fk_order_items_order_id_orders'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_order_items'))
    )
    op.create_index(op.f('ix_order_items_menu_item_id'), 'order_items', ['menu_item_id'], unique=False)
    op.create_index(op.f('ix_order_items_order_id'), 'order_items', ['order_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_order_items_order_id'), table_name='order_items')
    op.drop_index(op.f('ix_order_items_menu_item_id'), table_name='order_items')
    op.drop_table('order_items')
    op.drop_index('ix_orders_restaurant_id_created_at', table_name='orders')
    op.drop_table('orders')
    op.execute(sa.schema.DropSequence(sa.Sequence('order_number_seq')))
