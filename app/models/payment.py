"""Payment records.  DEMO MODE ONLY: no gateway, no card data, no real money.

Each order has exactly one payment (unique ``order_id``), created ``pending``
when the order is placed, for exactly the order total.  Status lifecycle::

    pending ──► succeeded ──► refunded      (paid order cancelled; demo refund)
       ├──────► failed                      (simulated decline)
       ├──────► expired                     (reservation timed out unpaid)
       └──────► cancelled                   (order cancelled before payment)

Status changes come only from the payment service, never from the model or a
customer's statement.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.order import Order


class PaymentStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


PAYMENT_TRANSITIONS: dict[PaymentStatus, frozenset[PaymentStatus]] = {
    PaymentStatus.PENDING: frozenset({PaymentStatus.SUCCEEDED, PaymentStatus.FAILED,
                                      PaymentStatus.EXPIRED, PaymentStatus.CANCELLED}),
    PaymentStatus.SUCCEEDED: frozenset({PaymentStatus.REFUNDED}),
    PaymentStatus.FAILED: frozenset(),
    PaymentStatus.EXPIRED: frozenset(),
    PaymentStatus.CANCELLED: frozenset(),
    PaymentStatus.REFUNDED: frozenset(),
}  # fmt: skip

class DemoOutcome(StrEnum):
    """Outcomes a demo payment simulation may apply."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


DEMO_MODE = "demo"
DEMO_NOTICE = "DEMO PAYMENT MODE: simulated payment, no real money was charged or refunded."

_STATUS_VALUES = ", ".join(f"'{s.value}'" for s in PaymentStatus)


class Payment(TimestampMixin, Base):
    __tablename__ = "payments"
    __table_args__ = (
        CheckConstraint(f"status IN ({_STATUS_VALUES})", name="status_valid"),
        CheckConstraint("amount_paise > 0", name="amount_paise_positive"),
        CheckConstraint(f"mode = '{DEMO_MODE}'", name="mode_demo_only"),
        CheckConstraint("currency ~ '^[A-Z]{3}$'", name="currency_format"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # One payment per order: retries can never create a second payment.
    order_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orders.id", ondelete="RESTRICT"), unique=True
    )
    amount_paise: Mapped[int] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(String(3))
    mode: Mapped[str] = mapped_column(String(10), default=DEMO_MODE, server_default=DEMO_MODE)
    status: Mapped[str] = mapped_column(
        String(20), default=PaymentStatus.PENDING.value, server_default=PaymentStatus.PENDING.value
    )
    failure_reason: Mapped[str | None] = mapped_column(String(200))
    succeeded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # any final state

    order: Mapped[Order] = relationship(back_populates="payment")

    def __repr__(self) -> str:
        return f"<Payment {self.status} {self.amount_paise}p ({self.mode})>"
