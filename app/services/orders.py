"""Order creation, pricing and status changes.

Pricing rules: prices come only from ``menu_items`` rows read (and locked) in
the same transaction that writes the order.  All money is integer paise;
``line_total = unit_price * quantity`` and, with no taxes or fees yet,
``total = subtotal = sum(line totals)``.

Idempotency: ``(restaurant_id, idempotency_key)`` is unique in the database.
A retry with the same key and an equivalent request returns the original
order (``created=False``); the same key with a different request is a
``ConflictError``.  Equivalence is judged by ``request_fingerprint``.

Stock and payment: placing an order reserves stock for its tracked items
(under row locks, in the same transaction) and creates its pending demo
payment; the reservation lapses after ``order_reservation_ttl_seconds``.  See
``cancel`` and ``expire_reservations`` for how stock is given back, and
``app.services.payments`` for how a payment confirms the order.

Logging never includes customer names or phone numbers.
"""

import hashlib
import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from app.config import settings
from app.models import (
    CancelReason,
    InventoryState,
    MenuItem,
    Order,
    OrderItem,
    OrderStatus,
    Payment,
    PaymentStatus,
)
from app.models.order import can_transition
from app.schemas.order import OrderCreate, OrderLineRequest
from app.services import inventory as inventory_service
from app.services.common import (
    ConflictError,
    NotFoundError,
    UnprocessableError,
    commit,
    constraint_message,
    constraint_name,
)
from app.services.inventory import InsufficientStockError
from app.services.restaurants import get_restaurant

logger = logging.getLogger(__name__)

_IDEMPOTENCY_CONSTRAINT = "uq_orders_restaurant_id_idempotency_key"


@dataclass(frozen=True)
class PricedLine:
    """A requested line priced from the database."""

    menu_item_id: uuid.UUID
    item_name: str
    unit_price_paise: int
    quantity: int

    @property
    def line_total_paise(self) -> int:
        return self.unit_price_paise * self.quantity


@dataclass(frozen=True)
class OrderQuote:
    """Server-side pricing of requested lines (in request order)."""

    lines: tuple[PricedLine, ...]
    subtotal_paise: int
    total_paise: int  # == subtotal_paise until taxes/fees exist

    @classmethod
    def from_lines(cls, lines: Sequence[PricedLine]) -> "OrderQuote":
        subtotal = sum(line.line_total_paise for line in lines)
        return cls(lines=tuple(lines), subtotal_paise=subtotal, total_paise=subtotal)


class QuoteChangedError(ConflictError):
    """Prices, names or items differ from the quote the customer confirmed."""

    def __init__(self, quote: OrderQuote) -> None:
        super().__init__("Prices or items changed since the order was proposed.")
        self.quote = quote


@dataclass(frozen=True)
class OrderCreationResult:
    order: Order
    created: bool  # False when an idempotent retry returned an existing order


def price_lines(
    menu_items: Mapping[uuid.UUID, MenuItem], lines: Sequence[OrderLineRequest]
) -> list[PricedLine]:
    """Price each requested line from the given menu items (integer paise)."""
    return [
        PricedLine(
            menu_item_id=line.menu_item_id,
            item_name=menu_items[line.menu_item_id].name,
            unit_price_paise=menu_items[line.menu_item_id].price_paise,
            quantity=line.quantity,
        )
        for line in lines
    ]


def request_fingerprint(data: OrderCreate) -> str:
    """SHA-256 of the normalized request, excluding the idempotency key.

    Line order does not matter; every other field does.
    """
    canonical = {
        "customer_name": data.customer_name,
        "customer_phone": data.customer_phone,
        "notes": data.notes,
        "items": sorted([str(line.menu_item_id), line.quantity] for line in data.items),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _find_by_key(session: Session, restaurant_id: uuid.UUID, key: str) -> Order | None:
    return session.scalar(
        select(Order)
        .where(Order.restaurant_id == restaurant_id, Order.idempotency_key == key)
        .options(selectinload(Order.items))
    )


def _replay(existing: Order, fingerprint: str) -> OrderCreationResult:
    if existing.request_hash != fingerprint:
        raise ConflictError("idempotency_key was already used for a different order request.")
    logger.info("Idempotent replay of order %s", existing.order_number)
    return OrderCreationResult(order=existing, created=False)


def _load_menu_items(
    session: Session, restaurant_id: uuid.UUID, ids: list[uuid.UUID], *, lock: bool
) -> dict[uuid.UUID, MenuItem]:
    """Load the requested items that belong to this restaurant (fresh from the DB).

    With ``lock=True`` the rows are locked FOR SHARE, blocking concurrent
    price/availability changes until this transaction ends; a change committed
    while we waited is seen because the rows are re-read (``populate_existing``).
    Locking in ``id`` order keeps concurrent orders from deadlocking.
    """
    stmt = (
        select(MenuItem)
        .where(MenuItem.restaurant_id == restaurant_id, MenuItem.id.in_(ids))
        .order_by(MenuItem.id)
        .execution_options(populate_existing=True)
    )
    if lock:
        stmt = stmt.with_for_update(read=True)
    return {item.id: item for item in session.scalars(stmt).all()}


def _line_errors(
    menu_items: Mapping[uuid.UUID, MenuItem],
    lines: Sequence[OrderLineRequest],
    *,
    stock: bool = False,
) -> list[dict]:
    """Missing, foreign or unavailable items; with ``stock=True`` also tracked
    items without enough available stock (an unlocked, advisory check)."""
    errors = []
    for index, line in enumerate(lines):
        item = menu_items.get(line.menu_item_id)
        loc = ["body", "items", index, "menu_item_id"]
        if item is None:
            errors.append({
                "type": "menu_item_not_found",
                "loc": loc,
                "msg": "Menu item not found on this restaurant's menu.",
                "input": str(line.menu_item_id),
            })
        elif not item.is_available:
            errors.append({
                "type": "menu_item_unavailable",
                "loc": loc,
                "msg": f"'{item.name}' is currently unavailable.",
                "input": str(line.menu_item_id),
            })
        elif stock and item.available_quantity is not None and item.available_quantity < line.quantity:
            available = item.available_quantity
            errors.append({
                "type": "insufficient_stock",
                "loc": ["body", "items", index, "quantity"],
                "msg": (f"'{item.name}' is out of stock." if available <= 0 else
                        f"Only {available} of '{item.name}' available (requested {line.quantity})."),
                "input": str(line.menu_item_id),
                "available": max(available, 0),
            })  # fmt: skip
    return errors


def _quote(
    session: Session,
    restaurant_id: uuid.UUID,
    lines: Sequence[OrderLineRequest],
    *,
    lock: bool,
) -> OrderQuote:
    menu_items = _load_menu_items(
        session, restaurant_id, [line.menu_item_id for line in lines], lock=lock
    )
    # Unlocked quotes check stock as advice; locked ones leave it to the
    # reservation, which re-reads stock under FOR UPDATE.
    if errors := _line_errors(menu_items, lines, stock=not lock):
        raise UnprocessableError(errors)
    return OrderQuote.from_lines(price_lines(menu_items, lines))


def quote_order(
    session: Session, restaurant_id: uuid.UUID, lines: Sequence[OrderLineRequest]
) -> OrderQuote:
    """Validate and price lines exactly as ``create_order`` would, without saving.

    Raises the same errors: ``NotFoundError`` (restaurant), ``ConflictError``
    (inactive restaurant) and ``UnprocessableError`` (missing, foreign or
    unavailable items, or not enough stock right now).  Nothing is locked or
    reserved; ``create_order`` re-prices and reserves under locks, so pass the
    quote as ``expected`` to detect changes in between.
    """
    restaurant = get_restaurant(session, restaurant_id)
    if not restaurant.is_active:
        raise ConflictError("Restaurant is inactive; orders cannot be placed.")
    return _quote(session, restaurant_id, lines, lock=False)


def create_order(
    session: Session,
    restaurant_id: uuid.UUID,
    data: OrderCreate,
    *,
    expected: OrderQuote | None = None,
) -> OrderCreationResult:
    """Validate, price, reserve stock for and persist an order, with its
    pending demo payment, in one transaction.

    If ``expected`` is given (a quote the customer agreed to) and current
    pricing differs from it, ``QuoteChangedError`` is raised before anything is
    written; the error carries the new quote.  ``InsufficientStockError``
    (409) if a tracked item lacks stock; nothing is written then either.
    """
    # 404 if missing; FOR SHARE so it cannot be deactivated mid-order.
    restaurant = get_restaurant(session, restaurant_id, lock=True)
    key = data.idempotency_key or f"auto-{uuid.uuid4().hex}"
    fingerprint = request_fingerprint(data)

    existing = _find_by_key(session, restaurant_id, key)
    if existing is not None:
        return _replay(existing, fingerprint)

    if not restaurant.is_active:
        raise ConflictError("Restaurant is inactive; orders cannot be placed.")

    quote = _quote(session, restaurant_id, data.items, lock=True)
    if expected is not None and quote.lines != expected.lines:
        raise QuoteChangedError(quote)

    priced = quote.lines
    subtotal = quote.subtotal_paise
    lines = [
        OrderItem(
            id=uuid.uuid4(),
            menu_item_id=line.menu_item_id,
            item_name=line.item_name,
            unit_price_paise=line.unit_price_paise,
            quantity=line.quantity,
            line_total_paise=line.line_total_paise,
        )
        for line in priced
    ]
    # Same (item_name, id) order the relationship uses when reloading.
    lines.sort(key=lambda i: (i.item_name, i.id))
    order = Order(
        id=uuid.uuid4(),
        restaurant_id=restaurant_id,
        customer_name=data.customer_name,
        customer_phone=data.customer_phone,
        notes=data.notes,
        idempotency_key=key,
        request_hash=fingerprint,
        status=OrderStatus.PENDING.value,
        subtotal_paise=subtotal,
        total_paise=quote.total_paise,  # no taxes, fees or discounts yet
        reservation_expires_at=datetime.now(UTC)
        + timedelta(seconds=settings.order_reservation_ttl_seconds),
        items=lines,
    )
    session.add(order)

    try:
        session.flush()  # a concurrent duplicate key fails here, before reserving
        wanted = {line.menu_item_id: (line.item_name, line.quantity) for line in priced}
        tracked = inventory_service.reserve(session, order.id, wanted)
        order.inventory_state = (InventoryState.RESERVED if tracked else InventoryState.NONE).value
        session.add(Payment(order_id=order.id, amount_paise=order.total_paise,
                            currency=restaurant.currency))  # fmt: skip
        session.commit()  # order, lines, reservation and payment, or nothing
    except InsufficientStockError:
        session.rollback()
        raise
    except IntegrityError as exc:
        session.rollback()
        if constraint_name(exc) == _IDEMPOTENCY_CONSTRAINT:
            # A concurrent request with the same key committed first.
            existing = _find_by_key(session, restaurant_id, key)
            if existing is not None:
                return _replay(existing, fingerprint)
        raise ConflictError(constraint_message(exc)) from exc

    logger.info(
        "Created order %s for restaurant %s (%d lines, total %d paise)",
        order.order_number, restaurant_id, len(priced), subtotal,
    )
    return OrderCreationResult(order=order, created=True)


def get_order(session: Session, order_id: uuid.UUID) -> Order:
    order = session.get(
        Order, order_id, options=[selectinload(Order.items), selectinload(Order.payment)]
    )
    if order is None:
        raise NotFoundError("Order not found.")
    return order


def lock_order(session: Session, order_id: uuid.UUID) -> Order:
    """Lock an order (FOR UPDATE) and re-read it, its lines and payment.

    Every state change of an order, its payment or its stock movements starts
    here, so they are applied one at a time against the latest state.
    """
    order = session.get(
        Order,
        order_id,
        with_for_update=True,
        populate_existing=True,
        options=[selectinload(Order.items), selectinload(Order.payment)],
    )
    if order is None:
        raise NotFoundError("Order not found.")
    return order


def tracked_lines(order: Order) -> dict[uuid.UUID, int]:
    return {line.menu_item_id: line.quantity for line in order.items if line.menu_item_id}


def apply_cancellation(session: Session, order: Order, reason: CancelReason) -> None:
    """Cancel a locked order: give back its stock and close its payment.

    Effects are decided by recorded state, so they happen exactly once:
    - reserved stock is released (unpaid orders);
    - sold stock is returned only if preparation had not started
      (``confirmed``); once preparing, the food is considered used;
    - a pending payment becomes ``expired`` / ``failed`` / ``cancelled``; a
      succeeded one becomes ``refunded`` (demo refund).
    Does not commit.
    """
    now = datetime.now(UTC)
    state = InventoryState(order.inventory_state)
    if state == InventoryState.RESERVED:
        inventory_service.release(session, order.id, tracked_lines(order))
        order.inventory_state = InventoryState.RELEASED.value
    elif state == InventoryState.SOLD and order.status == OrderStatus.CONFIRMED.value:
        inventory_service.give_back(session, order.id, tracked_lines(order))
        order.inventory_state = InventoryState.RETURNED.value

    payment = order.payment
    if payment is not None:
        if payment.status == PaymentStatus.PENDING.value:
            payment.status = {
                CancelReason.EXPIRED: PaymentStatus.EXPIRED,
                CancelReason.PAYMENT_FAILED: PaymentStatus.FAILED,
            }.get(reason, PaymentStatus.CANCELLED).value
            payment.closed_at = now
        elif payment.status == PaymentStatus.SUCCEEDED.value:
            payment.status = PaymentStatus.REFUNDED.value
            payment.closed_at = now

    order.status = OrderStatus.CANCELLED.value
    order.cancel_reason = reason.value
    order.reservation_expires_at = None
    logger.info("Order %s cancelled (%s)", order.order_number, reason.value)


def cancel_order(
    session: Session, order_id: uuid.UUID, reason: CancelReason, *, by_customer: bool = False
) -> Order:
    """Cancel an order.  Repeating it is a no-op that returns the order.

    Customers may cancel only before preparation starts (``pending`` or
    ``confirmed``); staff may cancel any order that is not completed.
    """
    order = lock_order(session, order_id)
    if order.status == OrderStatus.CANCELLED.value:
        session.rollback()  # release the lock; nothing to do
        return get_order(session, order_id)
    if not can_transition(order.status, OrderStatus.CANCELLED):
        raise ConflictError(f"Cannot cancel an order that is '{order.status}'.")
    if by_customer and order.status not in (OrderStatus.PENDING, OrderStatus.CONFIRMED):
        raise ConflictError("This order can no longer be cancelled: preparation has started.")
    apply_cancellation(session, order, reason)
    commit(session)
    return order


def reservation_expired(order: Order, now: datetime | None = None) -> bool:
    return (
        order.status == OrderStatus.PENDING.value
        and order.reservation_expires_at is not None
        and order.reservation_expires_at <= (now or datetime.now(UTC))
    )


def expire_reservations(session: Session, *, limit: int = 100) -> list[str]:
    """Cancel unpaid orders whose reservation has lapsed, releasing their stock.

    Safe to call at any time and from several processes at once: rows locked
    by another caller are skipped (``SKIP LOCKED``) and each order's effects
    are applied once.  Returns the expired order numbers.
    """
    now = datetime.now(UTC)
    ids = session.scalars(
        select(Order.id)
        .where(Order.status == OrderStatus.PENDING.value, Order.reservation_expires_at <= now)
        .order_by(Order.reservation_expires_at, Order.id)
        .limit(limit)
        .with_for_update(skip_locked=True)
    ).all()
    expired = []
    for order_id in ids:
        order = lock_order(session, order_id)  # already locked by us; re-read
        if reservation_expired(order, now):
            apply_cancellation(session, order, CancelReason.EXPIRED)
            expired.append(order.order_number)
    commit(session)
    if expired:
        logger.info("Expired %d unpaid reservations", len(expired))
    return expired


def list_orders(
    session: Session,
    restaurant_id: uuid.UUID,
    *,
    limit: int,
    offset: int,
    status: OrderStatus | None = None,
) -> tuple[list[Order], int]:
    """One page of a restaurant's orders, newest first, with their items."""
    get_restaurant(session, restaurant_id)  # 404 for unknown restaurants
    conditions = [Order.restaurant_id == restaurant_id]
    if status is not None:
        conditions.append(Order.status == status.value)
    total = session.scalar(select(func.count()).select_from(Order).where(*conditions))
    orders = session.scalars(
        select(Order)
        .where(*conditions)
        .order_by(Order.created_at.desc(), Order.id.desc())
        .limit(limit)
        .offset(offset)
        .options(selectinload(Order.items), selectinload(Order.payment))
    ).all()
    return list(orders), total or 0


def update_order_status(session: Session, order_id: uuid.UUID, new_status: OrderStatus) -> Order:
    """Apply a status change allowed by ``ALLOWED_TRANSITIONS`` (staff).

    Setting the current status again is a no-op.  The row is locked
    (``FOR UPDATE``) so concurrent changes are applied one at a time, each
    checked against the latest status.  ``pending -> confirmed`` happens only
    through a succeeded payment, and cancelling goes through
    ``apply_cancellation`` so stock and payment are settled.
    """
    order = lock_order(session, order_id)
    current = OrderStatus(order.status)
    if current == new_status:
        session.rollback()  # release the lock; nothing to do
        return get_order(session, order_id)
    if not can_transition(current, new_status):
        raise ConflictError(
            f"Cannot change order status from '{current.value}' to '{new_status.value}'."
        )
    if current == OrderStatus.PENDING and new_status == OrderStatus.CONFIRMED:
        raise ConflictError(
            "An order is confirmed only when its payment succeeds; "
            "use the demo payment endpoint instead."
        )
    if new_status == OrderStatus.CANCELLED:
        apply_cancellation(session, order, CancelReason.STAFF)
    else:
        order.status = new_status.value
    commit(session)
    logger.info("Order %s status %s -> %s", order.order_number, current.value, new_status.value)
    return order
