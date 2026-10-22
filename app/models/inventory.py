"""Inventory: per-item stock levels and an append-only movement ledger.

Stock model (one row per tracked menu item; items without a row are not
stock-tracked and limited only by ``menu_items.is_available``)::

    on_hand    physical units in the restaurant that are not yet sold
    reserved   units held for placed-but-unpaid orders (a subset of on_hand)
    available  = on_hand - reserved   (what new orders may take)
    sold       cumulative units sold through paid orders (reporting only)

Every change is one of these movements, applied under a row lock:

    restock / adjust  on_hand += n / on_hand = n      (staff)
    reserve           reserved += q                   (order placed)
    release           reserved -= q                   (unpaid order cancelled/expired/failed)
    sell              on_hand -= q, reserved -= q, sold += q   (payment succeeded)
    return            on_hand += q, sold -= q         (paid order cancelled before preparation)

``inventory_movements`` has a unique ``(order_id, menu_item_id, kind)``, so
the database itself refuses to apply any order movement twice.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.mixins import TimestampMixin


class MovementKind(StrEnum):
    RESTOCK = "restock"
    ADJUST = "adjust"
    RESERVE = "reserve"
    RELEASE = "release"
    SELL = "sell"
    RETURN = "return"


_KIND_VALUES = ", ".join(f"'{k.value}'" for k in MovementKind)


class InventoryItem(TimestampMixin, Base):
    __tablename__ = "inventory_items"
    __table_args__ = (
        CheckConstraint("on_hand >= 0", name="on_hand_nonnegative"),
        CheckConstraint("reserved >= 0", name="reserved_nonnegative"),
        # Reservations can never exceed physical stock: this is the database's
        # last line of defence against overselling.
        CheckConstraint("reserved <= on_hand", name="reserved_within_on_hand"),
        CheckConstraint("sold >= 0", name="sold_nonnegative"),
    )

    # One row per tracked menu item; removed together with the menu item.
    menu_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("menu_items.id", ondelete="CASCADE"), primary_key=True
    )
    on_hand: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    reserved: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    sold: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    @property
    def available(self) -> int:
        return self.on_hand - self.reserved

    def __repr__(self) -> str:
        return f"<InventoryItem on_hand={self.on_hand} reserved={self.reserved} sold={self.sold}>"


class InventoryMovement(Base):
    """Append-only audit ledger.  Rows are never updated or deleted."""

    __tablename__ = "inventory_movements"
    __table_args__ = (
        CheckConstraint(f"kind IN ({_KIND_VALUES})", name="kind_valid"),
        CheckConstraint("quantity >= 0", name="quantity_nonnegative"),
        # Each order applies each kind of movement to each item at most once.
        Index(
            "uq_inventory_movements_order_item_kind",
            "order_id", "menu_item_id", "kind",
            unique=True,
            postgresql_where=text("order_id IS NOT NULL"),
        ),  # fmt: skip
        Index("ix_inventory_movements_menu_item_id_created_at", "menu_item_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # SET NULL keeps the audit trail if a menu item is deleted.
    menu_item_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("menu_items.id", ondelete="SET NULL")
    )
    order_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orders.id", ondelete="RESTRICT"), index=True
    )
    kind: Mapped[str] = mapped_column(String(20))
    quantity: Mapped[int] = mapped_column(Integer)
    # Stock levels right after this movement, for auditing.
    on_hand_after: Mapped[int] = mapped_column(Integer)
    reserved_after: Mapped[int] = mapped_column(Integer)
    note: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.statement_timestamp()
    )

    menu_item = relationship("MenuItem")

    def __repr__(self) -> str:
        return f"<InventoryMovement {self.kind} {self.quantity}>"
