"""Order endpoints.

Public: placing an order, and (with that order's ``X-Order-Token``) tracking
or cancelling it.  Staff (``X-Staff-Key``): reading and listing orders, which
include customer contact details, and changing their status.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import CancelReason, OrderStatus
from app.schemas import OrderCreate, OrderRead, OrderStatusUpdate, Page
from app.schemas.common import MAX_PAGE_SIZE
from app.schemas.order import OrderCreatedRead, OrderTrackRead
from app.security import is_staff, order_token, require_staff, valid_order_token
from app.services import orders as service

router = APIRouter(prefix="/api/v1", tags=["orders"])

SessionDep = Annotated[Session, Depends(get_db)]
StaffOnly = [Depends(require_staff)]


def _authorize_order(order_id: uuid.UUID, token: str | None, staff_key: str | None) -> bool:
    """True for staff, False for the order's customer; 404 otherwise (an order
    that is not yours is indistinguishable from one that does not exist)."""
    if is_staff(staff_key):
        return True
    if valid_order_token(order_id, token):
        return False
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found.")


@router.post(
    "/restaurants/{restaurant_id}/orders",
    status_code=status.HTTP_201_CREATED,
    response_model=OrderCreatedRead,
    responses={200: {"description": "Idempotent retry: the existing order is returned."}},
)
def create_order(
    restaurant_id: uuid.UUID, body: OrderCreate, db: SessionDep, response: Response
):
    """Place an order.  Prices and totals are computed by the server.

    The order starts ``pending``: stock for its tracked items is reserved
    until ``reservation_expires_at`` and a DEMO payment for the exact total is
    created ``pending``.  It becomes ``confirmed`` only when that payment
    succeeds.  Keep ``customer_token`` to track or cancel the order.

    - **201** — order created.
    - **200** — same ``idempotency_key`` and same request as an earlier order:
      that order is returned unchanged, with header ``Idempotent-Replayed: true``.
    - **409** — ``idempotency_key`` reused with a different request, the
      restaurant is inactive, or not enough stock.
    - **422** — invalid body, or menu items that are missing, from another
      restaurant, or unavailable.
    """
    result = service.create_order(db, restaurant_id, body)
    if not result.created:
        response.status_code = status.HTTP_200_OK
        response.headers["Idempotent-Replayed"] = "true"
    order = OrderRead.model_validate(result.order)
    return OrderCreatedRead(**order.model_dump(), customer_token=order_token(result.order.id))


@router.get(
    "/restaurants/{restaurant_id}/orders", response_model=Page[OrderRead], dependencies=StaffOnly
)
def list_orders(
    restaurant_id: uuid.UUID,
    db: SessionDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    status_filter: Annotated[OrderStatus | None, Query(alias="status")] = None,
):
    """List a restaurant's orders, newest first, optionally filtered by status (staff)."""
    items, total = service.list_orders(
        db, restaurant_id, limit=limit, offset=offset, status=status_filter
    )
    return Page[OrderRead](items=items, total=total, limit=limit, offset=offset)


@router.get("/orders/{order_id}", response_model=OrderRead, dependencies=StaffOnly)
def get_order(order_id: uuid.UUID, db: SessionDep):
    """Fetch an order with its line items, price snapshots and payment (staff)."""
    return service.get_order(db, order_id)


@router.get("/orders/{order_id}/track", response_model=OrderTrackRead)
def track_order(
    order_id: uuid.UUID,
    db: SessionDep,
    x_order_token: Annotated[str | None, Header()] = None,
    x_staff_key: Annotated[str | None, Header()] = None,
):
    """A customer's view of their own order: status, items, total and payment
    status, without contact details.  Needs ``X-Order-Token`` (or staff)."""
    _authorize_order(order_id, x_order_token, x_staff_key)
    return service.get_order(db, order_id)


@router.post("/orders/{order_id}/cancel", response_model=OrderTrackRead)
def cancel_order(
    order_id: uuid.UUID,
    db: SessionDep,
    x_order_token: Annotated[str | None, Header()] = None,
    x_staff_key: Annotated[str | None, Header()] = None,
):
    """Cancel an order; repeating it changes nothing.  Reserved stock is
    released and a pending payment cancelled; a paid order is refunded (demo).

    Customers (``X-Order-Token``) may cancel before preparation starts; staff
    may cancel any order that is not completed.  409 if not allowed.
    """
    staff = _authorize_order(order_id, x_order_token, x_staff_key)
    reason = CancelReason.STAFF if staff else CancelReason.CUSTOMER
    return service.cancel_order(db, order_id, reason, by_customer=not staff)


@router.patch("/orders/{order_id}/status", response_model=OrderRead, dependencies=StaffOnly)
def update_order_status(order_id: uuid.UUID, body: OrderStatusUpdate, db: SessionDep):
    """Change an order's status (staff).  409 if the transition is not allowed.

    ``pending -> confirmed`` is refused: an order is confirmed by its payment.
    ``cancelled`` settles stock and payment like ``POST .../cancel``.
    """
    return service.update_order_status(db, order_id, body.status)
