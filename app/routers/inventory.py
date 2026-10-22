"""Stock endpoints and reservation maintenance (staff only)."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.common import MAX_PAGE_SIZE
from app.schemas.inventory import MovementRead, StockRead, StockUpdate
from app.security import require_staff
from app.services import inventory as service
from app.services import orders as order_service

router = APIRouter(prefix="/api/v1", tags=["inventory"], dependencies=[Depends(require_staff)])

SessionDep = Annotated[Session, Depends(get_db)]


class ExpiryResult(BaseModel):
    expired_orders: list[str]


@router.get("/restaurants/{restaurant_id}/inventory", response_model=list[StockRead])
def list_stock(restaurant_id: uuid.UUID, db: SessionDep):
    """Every menu item with on-hand, reserved, available and sold units."""
    return [StockRead.of(item) for item in service.list_stock(db, restaurant_id)]


@router.put("/menu-items/{item_id}/inventory", response_model=StockRead)
def set_stock(item_id: uuid.UUID, body: StockUpdate, db: SessionDep):
    """Set the physical count (``on_hand``); starts tracking an untracked item.
    409 if it would drop below the units reserved for unpaid orders."""
    return StockRead.of(service.set_stock(db, item_id, on_hand=body.on_hand, note=body.note))


@router.get("/menu-items/{item_id}/inventory/movements", response_model=list[MovementRead])
def list_movements(
    item_id: uuid.UUID,
    db: SessionDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
):
    """The item's stock ledger, newest first."""
    return service.list_movements(db, item_id, limit=limit, offset=offset)


@router.post("/maintenance/expire-reservations", response_model=ExpiryResult)
def expire_reservations(db: SessionDep):
    """Cancel unpaid orders whose reservation lapsed and release their stock.
    Safe to call at any time; the app also runs this periodically."""
    return ExpiryResult(expired_orders=order_service.expire_reservations(db))
