"""Menu item persistence operations."""

import re
import uuid

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models import InventoryItem, MenuItem
from app.schemas import MenuItemCreate, MenuItemUpdate
from app.services.common import ConflictError, NotFoundError, commit
from app.services.restaurants import get_restaurant


def get_menu_item(session: Session, item_id: uuid.UUID) -> MenuItem:
    item = session.get(MenuItem, item_id)
    if item is None:
        raise NotFoundError("Menu item not found.")
    return item


def list_menu_items(
    session: Session,
    restaurant_id: uuid.UUID,
    *,
    limit: int,
    offset: int,
    is_available: bool | None = None,
    search: str | None = None,
    category: str | None = None,
    in_stock: bool | None = None,
) -> tuple[list[MenuItem], int]:
    """Return one page of a restaurant's menu items and the total count.

    ``search`` matches name or description (case-insensitive substring);
    ``category`` matches exactly, ignoring case.  ``in_stock`` filters on
    stock: untracked items count as in stock, tracked ones need at least one
    available unit.
    """
    get_restaurant(session, restaurant_id)  # 404 for unknown restaurants
    conditions = [MenuItem.restaurant_id == restaurant_id]
    if is_available is not None:
        conditions.append(MenuItem.is_available.is_(is_available))
    if in_stock is not None:
        has_stock = (
            select(InventoryItem.menu_item_id)
            .where(InventoryItem.on_hand - InventoryItem.reserved > 0)
            .scalar_subquery()
        )
        tracked = select(InventoryItem.menu_item_id).scalar_subquery()
        stocked = or_(MenuItem.id.not_in(tracked), MenuItem.id.in_(has_stock))
        conditions.append(stocked if in_stock else ~stocked)
    if search:
        # Escape LIKE wildcards so the text is matched literally.
        pattern = "%" + re.sub(r"([\\%_])", r"\\\1", search) + "%"
        conditions.append(
            or_(
                MenuItem.name.ilike(pattern, escape="\\"),
                MenuItem.description.ilike(pattern, escape="\\"),
            )
        )
    if category:
        conditions.append(func.lower(MenuItem.category) == category.lower())
    total = session.scalar(select(func.count()).select_from(MenuItem).where(*conditions))
    items = session.scalars(
        select(MenuItem)
        .where(*conditions)
        .order_by(MenuItem.name, MenuItem.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return list(items), total or 0


def create_menu_item(
    session: Session, restaurant_id: uuid.UUID, data: MenuItemCreate
) -> MenuItem:
    # Lock the restaurant row so it cannot be deactivated mid-request.
    restaurant = get_restaurant(session, restaurant_id, lock=True)
    if not restaurant.is_active:
        raise ConflictError("Restaurant is inactive; menu items cannot be added.")
    item = MenuItem(**data.model_dump(), restaurant=restaurant)
    session.add(item)
    commit(session)
    return item


def update_menu_item(session: Session, item_id: uuid.UUID, data: MenuItemUpdate) -> MenuItem:
    item = get_menu_item(session, item_id)
    for field, value in data.changes().items():
        setattr(item, field, value)
    commit(session)
    return item


def delete_menu_item(session: Session, item_id: uuid.UUID) -> None:
    item = get_menu_item(session, item_id)
    if item.inventory is not None and item.inventory.reserved > 0:
        raise ConflictError(
            "This item has stock reserved for unpaid orders; switch it off "
            "(is_available=false) instead, or wait for those orders to finish."
        )
    session.delete(item)
    commit(session)
