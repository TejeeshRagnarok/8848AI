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
    include_inactive: bool = False,
) -> tuple[list[MenuItem], int]:
    """Return one page of a restaurant's menu items and the total count.

    ``search`` matches name or description (case-insensitive substring);
    ``category`` matches exactly, ignoring case.  ``in_stock`` filters on
    stock: untracked items count as in stock, tracked ones need at least one
    available unit.  Retired (inactive) items are left out unless
    ``include_inactive`` (staff views).
    """
    get_restaurant(session, restaurant_id)  # 404 for unknown restaurants
    conditions = [MenuItem.restaurant_id == restaurant_id]
    if not include_inactive:
        conditions.append(MenuItem.is_active.is_(True))
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
    _ensure_unique_name(session, restaurant_id, data.name)
    item = MenuItem(**data.model_dump(), restaurant=restaurant)
    session.add(item)
    commit(session)
    return item


def _ensure_unique_name(
    session: Session, restaurant_id: uuid.UUID, name: str, *, exclude: uuid.UUID | None = None
) -> None:
    """One active item per name (ignoring case) on a restaurant's menu, so
    the same dish is not added twice by accident.  Retired items do not count."""
    stmt = select(MenuItem.id).where(
        MenuItem.restaurant_id == restaurant_id,
        MenuItem.is_active.is_(True),
        func.lower(MenuItem.name) == name.strip().lower(),
    )
    if exclude is not None:
        stmt = stmt.where(MenuItem.id != exclude)
    if session.scalar(stmt.limit(1)) is not None:
        raise ConflictError(f"An active menu item named '{name.strip()}' already exists.")


def set_menu_item_active(session: Session, item_id: uuid.UUID, active: bool) -> MenuItem:
    """Retire (or bring back) a dish.  Nothing is deleted: past order lines,
    payments and the stock ledger keep pointing at it."""
    item = session.get(MenuItem, item_id, with_for_update=True, populate_existing=True)
    if item is None:
        raise NotFoundError("Menu item not found.")
    if item.is_active == active:
        session.rollback()
        return get_menu_item(session, item_id)
    if not active and item.inventory is not None and item.inventory.reserved > 0:
        raise ConflictError(
            "This item has stock reserved for unpaid orders; mark it unavailable instead, "
            "or wait for those orders to finish."
        )
    if active:
        _ensure_unique_name(session, item.restaurant_id, item.name, exclude=item.id)
    item.is_active = active
    commit(session)
    return item


def update_menu_item(session: Session, item_id: uuid.UUID, data: MenuItemUpdate) -> MenuItem:
    item = get_menu_item(session, item_id)
    changes = data.changes()
    if "name" in changes and item.is_active:
        _ensure_unique_name(session, item.restaurant_id, changes["name"], exclude=item.id)
    for field, value in changes.items():
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
