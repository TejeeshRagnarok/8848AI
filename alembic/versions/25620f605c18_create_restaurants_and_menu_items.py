"""create_restaurants_and_menu_items

Revision ID: 25620f605c18
Revises: 5900f7d42043
Create Date: 2026-10-22 11:55:45.685698

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '25620f605c18'
down_revision: Union[str, Sequence[str], None] = '5900f7d42043'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('restaurants',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('slug', sa.String(length=100), nullable=False),
    sa.Column('phone', sa.String(length=20), nullable=True),
    sa.Column('timezone', sa.String(length=64), server_default='Asia/Kolkata', nullable=False),
    sa.Column('currency', sa.String(length=3), server_default='INR', nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('statement_timestamp()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('statement_timestamp()'), nullable=False),
    sa.CheckConstraint("currency ~ '^[A-Z]{3}$'", name=op.f('ck_restaurants_currency_format')),
    sa.CheckConstraint("slug ~ '^[a-z0-9]+(-[a-z0-9]+)*$'", name=op.f('ck_restaurants_slug_format')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_restaurants')),
    sa.UniqueConstraint('slug', name=op.f('uq_restaurants_slug'))
    )
    op.create_table('menu_items',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('restaurant_id', sa.Uuid(), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('category', sa.String(length=100), nullable=True),
    sa.Column('price_paise', sa.Integer(), nullable=False),
    sa.Column('is_available', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('statement_timestamp()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('statement_timestamp()'), nullable=False),
    sa.CheckConstraint('price_paise > 0', name=op.f('ck_menu_items_price_paise_positive')),
    sa.ForeignKeyConstraint(['restaurant_id'], ['restaurants.id'], name=op.f('fk_menu_items_restaurant_id_restaurants'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_menu_items'))
    )
    op.create_index(op.f('ix_menu_items_restaurant_id'), 'menu_items', ['restaurant_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_menu_items_restaurant_id'), table_name='menu_items')
    op.drop_table('menu_items')
    op.drop_table('restaurants')
