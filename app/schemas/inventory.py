"""Schemas for stock levels and the movement ledger (staff endpoints)."""

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.schemas.common import RequestSchema


class StockUpdate(RequestSchema):
    """Set the physical count (starts tracking an untracked item)."""

    on_hand: Annotated[int, Field(strict=True, ge=0, le=1_000_000)]
    note: Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)] | None = None


class StockRead(BaseModel):
    menu_item_id: uuid.UUID
    name: str
    is_available: bool
    stock_tracked: bool
    on_hand: int | None
    reserved: int | None
    available: int | None
    sold: int | None
    orderable: bool

    @classmethod
    def of(cls, item) -> "StockRead":
        inv = item.inventory
        return cls(
            menu_item_id=item.id, name=item.name, is_available=item.is_available,
            stock_tracked=inv is not None,
            on_hand=inv.on_hand if inv else None, reserved=inv.reserved if inv else None,
            available=inv.available if inv else None, sold=inv.sold if inv else None,
            orderable=item.orderable,
        )  # fmt: skip


class MovementRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    menu_item_id: uuid.UUID | None
    order_id: uuid.UUID | None
    kind: str
    quantity: int
    on_hand_after: int
    reserved_after: int
    note: str | None
    created_at: datetime
