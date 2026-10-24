"""Request/response schemas for orders.

Clients send only menu item IDs and quantities.  Prices, line totals and
order totals are never accepted from the client; they are computed from the
database (unknown fields such as ``unit_price_paise`` are rejected).
"""

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from app.models.order import OrderStatus
from app.models.payment import DEMO_NOTICE
from app.schemas.common import Phone, RequestSchema, ShortText

MAX_LINES_PER_ORDER = 50
MAX_QUANTITY_PER_LINE = 100

Quantity = Annotated[int, Field(strict=True, ge=1, le=MAX_QUANTITY_PER_LINE)]
IdempotencyKey = Annotated[
    str, StringConstraints(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
]
Notes = Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)]


class OrderLineRequest(RequestSchema):
    menu_item_id: uuid.UUID
    quantity: Quantity


class OrderCreate(RequestSchema):
    customer_name: ShortText | None = None
    customer_phone: Phone
    items: Annotated[list[OrderLineRequest], Field(min_length=1, max_length=MAX_LINES_PER_ORDER)]
    notes: Notes | None = None
    # Send the same key when retrying so the order is created at most once.
    # If omitted, the server generates one (and retries cannot be deduplicated).
    idempotency_key: IdempotencyKey | None = None

    @field_validator("items")
    @classmethod
    def _no_duplicate_menu_items(cls, items: list[OrderLineRequest]) -> list[OrderLineRequest]:
        ids = [line.menu_item_id for line in items]
        if len(ids) != len(set(ids)):
            raise ValueError("each menu_item_id may appear only once; combine quantities instead")
        return items


class WalkInOrderCreate(OrderCreate):
    """A counter sale recorded by staff: the contact number is optional."""

    customer_phone: Phone | None = None


class OrderStatusUpdate(RequestSchema):
    status: OrderStatus


class PaymentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    amount_paise: int
    currency: str
    mode: str
    status: str
    simulated: bool = True
    notice: str = DEMO_NOTICE


class OrderItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    menu_item_id: uuid.UUID | None  # null if the menu item was later deleted
    item_name: str
    unit_price_paise: int
    quantity: int
    line_total_paise: int
    created_at: datetime


class OrderRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    restaurant_id: uuid.UUID
    order_number: str
    customer_name: str | None
    customer_phone: str | None  # absent only for walk-in sales
    status: OrderStatus
    subtotal_paise: int
    total_paise: int
    notes: str | None
    idempotency_key: str
    items: list[OrderItemRead]
    inventory_state: str = "none"
    source: str = "API"  # channel: AI_VOICE, AI_CHAT, WALK_IN_POS, API, UNKNOWN
    reservation_expires_at: datetime | None = None
    cancel_reason: str | None = None
    payment: PaymentSummary | None = None
    created_at: datetime
    updated_at: datetime


class OrderCreatedRead(OrderRead):
    # Proof of ownership for GET /orders/{id}/track and POST /orders/{id}/cancel
    # (header X-Order-Token).  Keep it private to the customer.
    customer_token: str


class OrderTrackRead(BaseModel):
    """What a customer may see about their own order (no contact details)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    order_number: str
    status: OrderStatus
    items: list[OrderItemRead]
    total_paise: int
    reservation_expires_at: datetime | None
    cancel_reason: str | None
    payment: PaymentSummary | None
    created_at: datetime
    updated_at: datetime
