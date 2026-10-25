"""Facts derived from order history, for the assistant's answers.

Popularity counts only paid orders (``confirmed`` and later; cancelled,
expired and unpaid orders are excluded) and only items still on the menu.
Below ``MIN_ORDERS_FOR_POPULARITY`` paid orders the result says so, so the
assistant never presents a guess as a best-seller.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import MenuItem, Order, OrderItem, OrderStatus
from app.services.restaurants import get_restaurant

MIN_ORDERS_FOR_POPULARITY = 5

PAID_STATUSES = (
    OrderStatus.CONFIRMED.value,
    OrderStatus.PREPARING.value,
    OrderStatus.READY.value,
    OrderStatus.COMPLETED.value,
)


@dataclass(frozen=True)
class PopularItem:
    menu_item: MenuItem
    units_sold: int
    orders: int  # paid orders containing the item


@dataclass(frozen=True)
class Popularity:
    paid_orders: int
    sufficient_data: bool
    items: list[PopularItem]  # most units sold first


def popular_items(session: Session, restaurant_id: uuid.UUID, *, limit: int = 3) -> Popularity:
    get_restaurant(session, restaurant_id)  # 404 for unknown restaurants
    paid = (Order.restaurant_id == restaurant_id, Order.status.in_(PAID_STATUSES))
    paid_orders = session.scalar(select(func.count()).select_from(Order).where(*paid)) or 0
    rows = session.execute(
        select(
            OrderItem.menu_item_id,
            func.sum(OrderItem.quantity).label("units"),
            func.count(func.distinct(OrderItem.order_id)).label("orders"),
        )
        .join(Order, Order.id == OrderItem.order_id)
        .where(*paid, OrderItem.menu_item_id.is_not(None))
        .group_by(OrderItem.menu_item_id)
        .order_by(func.sum(OrderItem.quantity).desc(), OrderItem.menu_item_id)
        .limit(limit)
    ).all()
    menu = {
        item.id: item
        for item in session.scalars(
            select(MenuItem).where(MenuItem.id.in_([r.menu_item_id for r in rows]))
        )
    }
    items = [
        PopularItem(menu_item=menu[r.menu_item_id], units_sold=int(r.units), orders=int(r.orders))
        for r in rows
        if r.menu_item_id in menu
    ]
    return Popularity(
        paid_orders=paid_orders,
        sufficient_data=paid_orders >= MIN_ORDERS_FOR_POPULARITY,
        items=items,
    )
