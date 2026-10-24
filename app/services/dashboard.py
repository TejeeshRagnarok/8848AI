"""Read models for the restaurant dashboard (orders, summary, change feed).

Everything is scoped to one restaurant id, which always comes from the
signed-in dashboard session, never from the browser.  State changes reuse
the existing order, payment, menu and inventory services.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.models import InventoryItem, MenuItem, Order, OrderSource, OrderStatus, Payment
from app.services.common import NotFoundError
from app.services.restaurants import get_restaurant

ACTIVE_STATUSES = (
    OrderStatus.PENDING.value,
    OrderStatus.CONFIRMED.value,
    OrderStatus.PREPARING.value,
    OrderStatus.READY.value,
)


def get_scoped_order(session: Session, restaurant_id: uuid.UUID, order_id: uuid.UUID) -> Order:
    """An order of this restaurant; another restaurant's order is 'not found'."""
    order = session.get(Order, order_id, options=[selectinload(Order.items), selectinload(Order.payment)])
    if order is None or order.restaurant_id != restaurant_id:
        raise NotFoundError("Order not found.")
    return order


def get_scoped_item(session: Session, restaurant_id: uuid.UUID, item_id: uuid.UUID) -> MenuItem:
    item = session.get(MenuItem, item_id)
    if item is None or item.restaurant_id != restaurant_id:
        raise NotFoundError("Menu item not found.")
    return item


def list_orders(
    session: Session,
    restaurant_id: uuid.UUID,
    *,
    status: OrderStatus | None = None,
    source: OrderSource | None = None,
    active_only: bool = False,
    search: str | None = None,
    limit: int,
    offset: int,
) -> tuple[list[Order], int]:
    """Newest first.  ``search`` matches the order number, customer name or
    the digits of the contact number."""
    conditions = [Order.restaurant_id == restaurant_id]
    if status is not None:
        conditions.append(Order.status == status.value)
    if source is not None:
        conditions.append(Order.source == source.value)
    if active_only:
        conditions.append(Order.status.in_(ACTIVE_STATUSES))
    if search:
        text = search.strip()
        like = "%" + re.sub(r"([\\%_])", r"\\\1", text) + "%"
        matches = [Order.order_number.ilike(like, escape="\\"), Order.customer_name.ilike(like, escape="\\")]
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 3:
            matches.append(func.regexp_replace(Order.customer_phone, r"\D", "", "g").contains(digits))
        conditions.append(or_(*matches))
    total = session.scalar(select(func.count()).select_from(Order).where(*conditions)) or 0
    orders = session.scalars(
        select(Order)
        .where(*conditions)
        .order_by(Order.created_at.desc(), Order.id.desc())
        .limit(limit)
        .offset(offset)
        .options(selectinload(Order.items), selectinload(Order.payment))
    ).all()
    return list(orders), total


@dataclass(frozen=True)
class Summary:
    restaurant_name: str
    currency: str
    active_orders: int
    orders_today: int
    awaiting_payment: int
    low_stock: int
    out_of_stock: int
    unavailable: int


def summary(session: Session, restaurant_id: uuid.UUID, *, low_stock_threshold: int) -> Summary:
    restaurant = get_restaurant(session, restaurant_id)
    tz = ZoneInfo(restaurant.timezone)
    start_of_day = datetime.combine(datetime.now(tz).date(), time.min, tzinfo=tz).astimezone(UTC)
    mine = Order.restaurant_id == restaurant_id

    def count(*conditions) -> int:
        return session.scalar(select(func.count()).select_from(Order).where(mine, *conditions)) or 0

    items = session.scalars(
        select(MenuItem).where(MenuItem.restaurant_id == restaurant_id, MenuItem.is_active.is_(True))
    ).all()
    tracked = [i for i in items if i.inventory is not None]
    return Summary(
        restaurant_name=restaurant.name,
        currency=restaurant.currency,
        active_orders=count(Order.status.in_(ACTIVE_STATUSES)),
        orders_today=count(Order.created_at >= start_of_day),
        awaiting_payment=count(Order.status == OrderStatus.PENDING.value),
        low_stock=sum(1 for i in tracked if 0 < i.inventory.available <= low_stock_threshold),
        out_of_stock=sum(1 for i in tracked if i.inventory.available <= 0),
        unavailable=sum(1 for i in items if not i.is_available),
    )


def change_fingerprint(session: Session, restaurant_id: uuid.UUID) -> str:
    """Cheap token that changes whenever this restaurant's orders, payments,
    menu or stock change (every write updates ``updated_at`` or adds a row)."""
    order_part = session.execute(
        select(func.count(), func.max(Order.updated_at)).where(Order.restaurant_id == restaurant_id)
    ).one()
    menu_part = session.execute(
        select(func.count(), func.max(MenuItem.updated_at), func.max(InventoryItem.updated_at))
        .select_from(MenuItem)
        .outerjoin(InventoryItem, InventoryItem.menu_item_id == MenuItem.id)
        .where(MenuItem.restaurant_id == restaurant_id)
    ).one()
    payment_part = session.scalar(
        select(func.max(Payment.updated_at)).join(Order, Order.id == Payment.order_id)
        .where(Order.restaurant_id == restaurant_id)
    )  # fmt: skip
    return "|".join(str(v) for v in (*order_part, *menu_part, payment_part))
