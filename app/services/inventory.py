"""Stock levels, reservations and the movement ledger.

See ``app.models.inventory`` for the stock model.  The order-side helpers
(``reserve`` / ``release`` / ``sell`` / ``give_back``) never commit: they run
inside the caller's transaction, after the caller has locked the order, so a
movement and the order state that records it are written together or not at
all.

Lock order (prevents deadlocks): orders → menu_items (FOR SHARE) →
inventory_items (FOR UPDATE), each set in primary-key order.
"""

import logging
import uuid
from collections.abc import Iterable, Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import InventoryItem, InventoryMovement, MenuItem, MovementKind
from app.services.common import ConflictError, NotFoundError, commit
from app.services.restaurants import get_restaurant

logger = logging.getLogger(__name__)


class InsufficientStockError(ConflictError):
    """Not enough available stock.  ``shortages`` maps item id -> (name, requested, available)."""

    def __init__(self, shortages: dict[uuid.UUID, tuple[str, int, int]]) -> None:
        parts = [f"only {avail} of '{name}' available (requested {want})"
                 for name, want, avail in shortages.values()]  # fmt: skip
        super().__init__("Not enough stock: " + "; ".join(parts) + ".")
        self.shortages = shortages


def lock_inventory(session: Session, menu_item_ids: Iterable[uuid.UUID]) -> dict[uuid.UUID, InventoryItem]:
    """Lock (FOR UPDATE, in id order) and return the inventory rows of tracked items."""
    ids = sorted(set(menu_item_ids))
    if not ids:
        return {}
    rows = session.scalars(
        select(InventoryItem)
        .where(InventoryItem.menu_item_id.in_(ids))
        .order_by(InventoryItem.menu_item_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).all()
    return {row.menu_item_id: row for row in rows}


def _record(
    session: Session,
    row: InventoryItem,
    kind: MovementKind,
    quantity: int,
    *,
    order_id: uuid.UUID | None = None,
    note: str | None = None,
) -> None:
    session.add(
        InventoryMovement(
            menu_item_id=row.menu_item_id,
            order_id=order_id,
            kind=kind.value,
            quantity=quantity,
            on_hand_after=row.on_hand,
            reserved_after=row.reserved,
            note=note,
        )
    )


# ── Order-side movements (caller holds the order lock; no commit here) ──────


def check_stock(
    rows: Mapping[uuid.UUID, InventoryItem], wanted: Mapping[uuid.UUID, tuple[str, int]]
) -> None:
    """Raise ``InsufficientStockError`` if any tracked item lacks available stock.

    ``wanted`` maps menu item id -> (name, quantity); untracked items pass.
    """
    shortages = {
        item_id: (name, qty, rows[item_id].available)
        for item_id, (name, qty) in wanted.items()
        if item_id in rows and rows[item_id].available < qty
    }
    if shortages:
        raise InsufficientStockError(shortages)


def reserve(
    session: Session, order_id: uuid.UUID, lines: Mapping[uuid.UUID, tuple[str, int]]
) -> bool:
    """Reserve stock for an order's tracked lines.  Returns whether any were tracked."""
    rows = lock_inventory(session, lines)
    check_stock(rows, lines)
    for item_id, row in rows.items():
        row.reserved += lines[item_id][1]
        _record(session, row, MovementKind.RESERVE, lines[item_id][1], order_id=order_id)
    return bool(rows)


def release(session: Session, order_id: uuid.UUID, lines: Mapping[uuid.UUID, int]) -> None:
    """Give a reservation back (unpaid order cancelled, failed or expired)."""
    for item_id, row in lock_inventory(session, lines).items():
        row.reserved -= lines[item_id]
        _record(session, row, MovementKind.RELEASE, lines[item_id], order_id=order_id)


def sell(session: Session, order_id: uuid.UUID, lines: Mapping[uuid.UUID, int]) -> None:
    """Convert a reservation into a sale (payment succeeded)."""
    for item_id, row in lock_inventory(session, lines).items():
        row.reserved -= lines[item_id]
        row.on_hand -= lines[item_id]
        row.sold += lines[item_id]
        _record(session, row, MovementKind.SELL, lines[item_id], order_id=order_id)


def give_back(session: Session, order_id: uuid.UUID, lines: Mapping[uuid.UUID, int]) -> None:
    """Put sold stock back (paid order cancelled before preparation began)."""
    for item_id, row in lock_inventory(session, lines).items():
        row.on_hand += lines[item_id]
        row.sold -= lines[item_id]
        _record(session, row, MovementKind.RETURN, lines[item_id], order_id=order_id)


# ── Staff operations ─────────────────────────────────────────────────────────


def set_stock(
    session: Session, menu_item_id: uuid.UUID, *, on_hand: int, note: str | None = None
) -> MenuItem:
    """Start tracking an item, or set its physical stock count (a stock-take).

    Reserved units stay reserved, so ``on_hand`` may not drop below them.
    """
    item = session.get(MenuItem, menu_item_id)
    if item is None:
        raise NotFoundError("Menu item not found.")
    row = lock_inventory(session, [menu_item_id]).get(menu_item_id)
    if row is None:
        row = InventoryItem(menu_item_id=menu_item_id, on_hand=on_hand, reserved=0, sold=0)
        session.add(row)
        session.flush()
        _record(session, row, MovementKind.RESTOCK, on_hand, note=note or "tracking started")
    else:
        if on_hand < row.reserved:
            raise ConflictError(
                f"on_hand cannot be below the {row.reserved} units reserved for unpaid orders."
            )
        change = on_hand - row.on_hand
        row.on_hand = on_hand
        kind = MovementKind.RESTOCK if change > 0 else MovementKind.ADJUST
        _record(session, row, kind, abs(change), note=note)
    commit(session)
    session.refresh(item)
    logger.info("Stock for menu item %s set to %d on hand", menu_item_id, on_hand)
    return item


def list_stock(session: Session, restaurant_id: uuid.UUID) -> list[MenuItem]:
    """All of a restaurant's menu items with their stock (tracked or not)."""
    get_restaurant(session, restaurant_id)
    return list(
        session.scalars(
            select(MenuItem)
            .where(MenuItem.restaurant_id == restaurant_id)
            .order_by(MenuItem.name, MenuItem.id)
            .execution_options(populate_existing=True)
        ).all()
    )


def list_movements(
    session: Session, menu_item_id: uuid.UUID, *, limit: int, offset: int
) -> list[InventoryMovement]:
    if session.get(MenuItem, menu_item_id) is None:
        raise NotFoundError("Menu item not found.")
    return list(
        session.scalars(
            select(InventoryMovement)
            .where(InventoryMovement.menu_item_id == menu_item_id)
            .order_by(InventoryMovement.created_at.desc(), InventoryMovement.id)
            .limit(limit)
            .offset(offset)
        ).all()
    )
