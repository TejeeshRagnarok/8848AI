"""DEMO payment service: simulated outcomes only, no gateway, no real money.

A payment is created (``pending``) together with its order.  Staff (or a
test harness holding the staff key) simulate the gateway's answer:

- ``succeeded``: the order becomes ``confirmed`` and its reserved stock is
  sold, in one transaction;
- ``failed``: the order is cancelled (``payment_failed``) and its stock
  released.

Rules: the simulated amount must equal the payment amount (which equals the
order total); repeating the same outcome returns the same result without
side effects; a payment that is already final cannot change outcome; a
reservation that has expired is expired instead of being paid.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import CancelReason, InventoryState, OrderStatus, Payment, PaymentStatus
from app.models.payment import DemoOutcome
from app.services import inventory as inventory_service
from app.services import orders as order_service
from app.services.common import ConflictError, NotFoundError, UnprocessableError, commit

logger = logging.getLogger(__name__)


class ReservationExpiredError(ConflictError):
    def __init__(self) -> None:
        super().__init__(
            "The order's reservation expired before payment; the order was cancelled and "
            "its stock released."
        )


@dataclass(frozen=True)
class SimulationResult:
    payment: Payment
    replayed: bool  # True when this outcome had already been applied


def get_payment(session: Session, payment_id: uuid.UUID) -> Payment:
    payment = session.get(Payment, payment_id)
    if payment is None:
        raise NotFoundError("Payment not found.")
    return payment


def get_order_payment(session: Session, order_id: uuid.UUID) -> Payment:
    order_service.get_order(session, order_id)  # 404 for unknown orders
    payment = session.scalar(select(Payment).where(Payment.order_id == order_id))
    if payment is None:
        raise NotFoundError("This order has no payment record.")
    return payment


def simulate(
    session: Session,
    payment_id: uuid.UUID,
    outcome: DemoOutcome,
    amount_paise: int,
    *,
    reason: str | None = None,
) -> SimulationResult:
    """Apply a simulated gateway outcome to a pending payment."""
    order_id = get_payment(session, payment_id).order_id
    order = order_service.lock_order(session, order_id)  # order first: see lock order
    payment = order.payment
    if payment is None or payment.id != payment_id:
        raise NotFoundError("Payment not found.")

    if amount_paise != payment.amount_paise:
        session.rollback()
        raise UnprocessableError([{
            "type": "amount_mismatch",
            "loc": ["body", "amount_paise"],
            "msg": f"Amount must equal the payment amount of {payment.amount_paise} paise.",
        }])  # fmt: skip

    if payment.status == outcome.value:
        session.rollback()
        return SimulationResult(payment=get_payment(session, payment_id), replayed=True)
    if payment.status != PaymentStatus.PENDING.value:
        raise ConflictError(
            f"Payment is already '{payment.status}'; it cannot be marked '{outcome.value}'."
        )
    if order.status != OrderStatus.PENDING.value:
        raise ConflictError(f"The order is '{order.status}' and cannot be paid.")
    if payment.amount_paise != order.total_paise:  # defensive: amounts are set together
        raise ConflictError("Payment amount does not match the order total.")

    if order_service.reservation_expired(order):
        order_service.apply_cancellation(session, order, CancelReason.EXPIRED)
        commit(session)
        raise ReservationExpiredError()

    if outcome == DemoOutcome.SUCCEEDED:
        if order.inventory_state == InventoryState.RESERVED.value:
            inventory_service.sell(session, order.id, order_service.tracked_lines(order))
            order.inventory_state = InventoryState.SOLD.value
        now = datetime.now(UTC)
        payment.status = PaymentStatus.SUCCEEDED.value
        payment.succeeded_at = now
        order.status = OrderStatus.CONFIRMED.value
        order.reservation_expires_at = None
    else:
        payment.failure_reason = reason or "Simulated payment failure."
        order_service.apply_cancellation(session, order, CancelReason.PAYMENT_FAILED)
    commit(session)
    logger.info("Demo payment for order %s %s", order.order_number, outcome.value)
    return SimulationResult(payment=payment, replayed=False)
