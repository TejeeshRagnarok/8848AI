"""Order and OrderItem ORM models, plus the order status lifecycle.

Status lifecycle (terminal states have no outgoing transitions)::

    pending ──► confirmed ──► preparing ──► ready ──► completed
       │            │             │           │
       └────────────┴─────────────┴───────────┴──► cancelled

- ``pending``: placed; stock reserved until ``reservation_expires_at``; the
  (demo) payment is pending.  Before this, a cart is only an unsaved proposal.
- ``confirmed``: reached only when the order's payment succeeds.
- ``cancelled``: ``cancel_reason`` says why (customer, staff,
  payment_failed, expired).

``inventory_state`` records which stock movement the order holds, so each is
applied exactly once: ``none`` (no tracked items) · ``reserved`` · ``sold``
· ``released`` (reservation given back) · ``returned`` (sold stock put back
after an early cancellation).  Delivery tracking is deliberately absent.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Sequence,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.payment import Payment
    from app.models.restaurant import Restaurant


class OrderStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PREPARING = "preparing"
    READY = "ready"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# Allowed status changes.  Any change not listed here is rejected.
ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.PENDING: frozenset({OrderStatus.CONFIRMED, OrderStatus.CANCELLED}),
    OrderStatus.CONFIRMED: frozenset({OrderStatus.PREPARING, OrderStatus.CANCELLED}),
    OrderStatus.PREPARING: frozenset({OrderStatus.READY, OrderStatus.CANCELLED}),
    OrderStatus.READY: frozenset({OrderStatus.COMPLETED, OrderStatus.CANCELLED}),
    OrderStatus.COMPLETED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
}


def can_transition(current: OrderStatus | str, new: OrderStatus | str) -> bool:
    return OrderStatus(new) in ALLOWED_TRANSITIONS[OrderStatus(current)]


class InventoryState(StrEnum):
    NONE = "none"
    RESERVED = "reserved"
    SOLD = "sold"
    RELEASED = "released"
    RETURNED = "returned"


class CancelReason(StrEnum):
    CUSTOMER = "customer"
    STAFF = "staff"
    PAYMENT_FAILED = "payment_failed"
    EXPIRED = "expired"


# Human-readable order numbers ("ORD-100001") from a PostgreSQL sequence:
# unique under concurrency, short enough to read out over the phone.
order_number_seq = Sequence("order_number_seq", start=100001, metadata=Base.metadata)

_STATUS_VALUES = ", ".join(f"'{s.value}'" for s in OrderStatus)
_INVENTORY_VALUES = ", ".join(f"'{s.value}'" for s in InventoryState)
_CANCEL_VALUES = ", ".join(f"'{s.value}'" for s in CancelReason)


class Order(TimestampMixin, Base):
    """A customer's order at one restaurant."""

    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint(f"status IN ({_STATUS_VALUES})", name="status_valid"),
        CheckConstraint("subtotal_paise >= 0", name="subtotal_paise_nonnegative"),
        CheckConstraint("total_paise >= 0", name="total_paise_nonnegative"),
        CheckConstraint(f"inventory_state IN ({_INVENTORY_VALUES})", name="inventory_state_valid"),
        CheckConstraint(
            f"cancel_reason IS NULL OR cancel_reason IN ({_CANCEL_VALUES})", name="cancel_reason_valid"
        ),
        CheckConstraint(
            "(status = 'cancelled') = (cancel_reason IS NOT NULL)", name="cancel_reason_iff_cancelled"
        ),
        # Serves the reservation-expiry sweep.
        Index(
            "ix_orders_pending_reservation_expires_at",
            "reservation_expires_at",
            postgresql_where=text("status = 'pending'"),
        ),
        # Idempotency keys are scoped to a restaurant; this also indexes lookups.
        UniqueConstraint(
            "restaurant_id", "idempotency_key", name="uq_orders_restaurant_id_idempotency_key"
        ),
        # Serves "list a restaurant's orders, newest first" and the FK.
        Index("ix_orders_restaurant_id_created_at", "restaurant_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    restaurant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("restaurants.id", ondelete="RESTRICT")
    )
    order_number: Mapped[str] = mapped_column(
        String(20),
        unique=True,
        server_default=text("'ORD-' || nextval('order_number_seq')"),
    )
    customer_name: Mapped[str | None] = mapped_column(String(100))
    customer_phone: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(
        String(20), default=OrderStatus.PENDING.value, server_default=OrderStatus.PENDING.value
    )
    subtotal_paise: Mapped[int] = mapped_column(BigInteger)
    total_paise: Mapped[int] = mapped_column(BigInteger)
    notes: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str] = mapped_column(String(100))
    # SHA-256 of the normalized creation request; detects key reuse with a
    # different request.
    request_hash: Mapped[str] = mapped_column(String(64))
    inventory_state: Mapped[str] = mapped_column(
        String(20), default=InventoryState.NONE.value, server_default=InventoryState.NONE.value
    )
    # When an unpaid order's stock reservation lapses (null once not pending).
    reservation_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_reason: Mapped[str | None] = mapped_column(String(30))

    restaurant: Mapped[Restaurant] = relationship()
    payment: Mapped[Payment | None] = relationship(back_populates="order", uselist=False)
    items: Mapped[list[OrderItem]] = relationship(
        back_populates="order",
        order_by="(OrderItem.item_name, OrderItem.id)",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Order {self.order_number!r} {self.status}>"


class OrderItem(Base):
    """One line of an order.  Name and price are snapshots taken at ordering
    time, so later menu edits never change order history.  Lines are never
    updated after creation (hence no ``updated_at``).
    """

    __tablename__ = "order_items"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="quantity_positive"),
        CheckConstraint("unit_price_paise > 0", name="unit_price_paise_positive"),
        CheckConstraint(
            "line_total_paise = unit_price_paise * quantity", name="line_total_matches"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Lines belong to their order; they are removed only with it.
    order_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"), index=True
    )
    # Nullable + SET NULL: deleting a menu item keeps order history intact
    # (the name/price snapshot remains); the link to the menu is just cleared.
    menu_item_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("menu_items.id", ondelete="SET NULL"), index=True
    )
    item_name: Mapped[str] = mapped_column(String(200))
    unit_price_paise: Mapped[int] = mapped_column(Integer)
    quantity: Mapped[int] = mapped_column(Integer)
    line_total_paise: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.statement_timestamp()
    )

    order: Mapped[Order] = relationship(back_populates="items")

    def __repr__(self) -> str:
        return f"<OrderItem {self.quantity} x {self.item_name!r}>"
