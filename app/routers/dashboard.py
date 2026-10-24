"""Restaurant dashboard API: live orders, and menu & live inventory.

Authentication: a staff member signs in once with the staff key; the browser
then holds only a signed HttpOnly session cookie bound to one restaurant (see
``app.security``).  Every endpoint takes the restaurant from that session,
never from the request, and every change goes through the same validated
services the voice agent and public API use (row locks, stock ledger,
payment and cancellation rules).  Payments here are DEMO only.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, get_session_factory
from app.models import CancelReason, MenuItem, OrderSource, OrderStatus, PaymentStatus, Restaurant
from app.models.payment import DemoOutcome
from app.schemas.common import MAX_PAGE_SIZE, RequestSchema
from app.schemas.inventory import MovementRead
from app.schemas.menu_item import MenuItemCreate, MenuItemRead, MenuItemUpdate
from app.schemas.order import OrderRead, OrderStatusUpdate, WalkInOrderCreate
from app.security import (
    DASHBOARD_COOKIE,
    DASHBOARD_COOKIE_PATH,
    DashboardSession,
    is_staff,
    issue_dashboard_session,
    login_throttle,
    read_dashboard_session,
    require_dashboard,
)
from app.services import dashboard as service
from app.services import inventory as inventory_service
from app.services import menu_items as menu_service
from app.services import orders as order_service
from app.services import payments as payment_service
from app.services.common import ConflictError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])

SessionDep = Annotated[Session, Depends(get_db)]
StaffDep = Annotated[DashboardSession, Depends(require_dashboard)]


# ── Schemas ──────────────────────────────────────────────────────────────────


class LoginRequest(RequestSchema):
    staff_key: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    # Which restaurant to manage; defaults to EXOTEL_RESTAURANT_ID, or the only
    # active restaurant.
    restaurant_slug: Annotated[str, StringConstraints(max_length=100)] | None = None


class SessionInfo(BaseModel):
    restaurant_id: uuid.UUID
    restaurant_name: str
    currency: str
    timezone: str
    expires_at: int


class SummaryRead(BaseModel):
    restaurant_name: str
    currency: str
    active_orders: int
    orders_today: int
    awaiting_payment: int
    low_stock: int
    out_of_stock: int
    unavailable: int
    low_stock_threshold: int


class DashboardItem(MenuItemRead):
    on_hand: int | None = None
    reserved: int | None = None
    sold: int | None = None


class NewMenuItem(MenuItemCreate):
    # Optional starting stock (tracked from the start, recorded in the ledger).
    initial_stock: Annotated[int, Field(strict=True, ge=0, le=1_000_000)] | None = None


class StockChange(RequestSchema):
    # Positive: delivery or correction up; negative: waste, breakage, count down.
    change: Annotated[int, Field(strict=True, ge=-1_000_000, le=1_000_000)]
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=200)]


class OrderPage(BaseModel):
    items: list[OrderRead]
    total: int
    limit: int
    offset: int


def _item(item: MenuItem) -> DashboardItem:
    inv = item.inventory
    return DashboardItem.model_validate(item).model_copy(update={
        "on_hand": inv.on_hand if inv else None,
        "reserved": inv.reserved if inv else None,
        "sold": inv.sold if inv else None,
    })  # fmt: skip


# ── Session ──────────────────────────────────────────────────────────────────


def _pick_restaurant(db: Session, slug: str | None) -> Restaurant:
    if slug:
        restaurant = db.scalar(select(Restaurant).where(Restaurant.slug == slug))
    elif settings.exotel_restaurant_id:
        restaurant = db.get(Restaurant, settings.exotel_restaurant_id)
    else:
        active = db.scalars(select(Restaurant).where(Restaurant.is_active.is_(True)).limit(2)).all()
        if len(active) != 1:
            raise HTTPException(status_code=422, detail="Specify restaurant_slug.")
        restaurant = active[0]
    if restaurant is None or not restaurant.is_active:
        raise HTTPException(status_code=404, detail="Restaurant not found.")
    return restaurant


def _info(restaurant: Restaurant, expires_at: int) -> SessionInfo:
    return SessionInfo(restaurant_id=restaurant.id, restaurant_name=restaurant.name,
                       currency=restaurant.currency, timezone=restaurant.timezone,
                       expires_at=expires_at)  # fmt: skip


@router.post("/session", response_model=SessionInfo)
def sign_in(body: LoginRequest, request: Request, response: Response, db: SessionDep):
    """Sign in with the staff key.  Sets an HttpOnly session cookie; the key
    itself is not stored in the browser.  5 failures per 5 minutes per client
    are allowed."""
    if settings.staff_api_key is None:
        raise HTTPException(status_code=503, detail="The dashboard is disabled until STAFF_API_KEY is configured.")
    client = request.client.host if request.client else "unknown"
    if login_throttle.blocked(client):
        raise HTTPException(status_code=429, detail="Too many failed sign-ins; try again in a few minutes.")
    if not is_staff(body.staff_key):
        login_throttle.failed(client)
        logger.warning("Dashboard sign-in failed")
        raise HTTPException(status_code=401, detail="Invalid staff key.")
    login_throttle.clear(client)
    restaurant = _pick_restaurant(db, body.restaurant_slug)
    token, lifetime = issue_dashboard_session(restaurant.id)
    response.set_cookie(
        DASHBOARD_COOKIE, token, max_age=lifetime, httponly=True, samesite="strict",
        secure=request.url.scheme == "https", path=DASHBOARD_COOKIE_PATH,
    )  # fmt: skip
    logger.info("Dashboard sign-in for restaurant %s", restaurant.id)
    return _info(restaurant, read_dashboard_session(token).expires_at)


@router.get("/session", response_model=SessionInfo)
def who_am_i(staff: StaffDep, db: SessionDep):
    restaurant = db.get(Restaurant, staff.restaurant_id)
    if restaurant is None:
        raise HTTPException(status_code=401, detail="Please sign in.")
    return _info(restaurant, staff.expires_at)


@router.delete("/session", status_code=status.HTTP_204_NO_CONTENT)
def sign_out(response: Response) -> Response:
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.delete_cookie(DASHBOARD_COOKIE, path=DASHBOARD_COOKIE_PATH)
    return response


# ── Summary and live updates ─────────────────────────────────────────────────


@router.get("/summary", response_model=SummaryRead)
def get_summary(staff: StaffDep, db: SessionDep):
    s = service.summary(db, staff.restaurant_id, low_stock_threshold=settings.dashboard_low_stock_threshold)
    return SummaryRead(**s.__dict__, low_stock_threshold=settings.dashboard_low_stock_threshold)


def _fingerprint(restaurant_id: uuid.UUID) -> str:
    with get_session_factory()() as db:
        return service.change_fingerprint(db, restaurant_id)


@router.get("/events")
async def events(request: Request, staff: StaffDep) -> StreamingResponse:
    """Server-Sent Events: ``change`` whenever this restaurant's orders,
    payments, menu or stock change.  The server checks one small aggregate
    query every DASHBOARD_EVENTS_SECONDS; the browser then refetches."""

    async def stream() -> AsyncIterator[str]:
        last, idle = None, 0.0
        yield "retry: 5000\n\n"
        while not await request.is_disconnected():
            try:
                current = await asyncio.to_thread(_fingerprint, staff.restaurant_id)
            except Exception:
                logger.exception("Dashboard change check failed")
                current = last
            if current != last:
                if last is not None:
                    yield "event: change\ndata: {}\n\n"
                last, idle = current, 0.0
            elif idle >= 15:
                yield ": keep-alive\n\n"
                idle = 0.0
            await asyncio.sleep(settings.dashboard_events_seconds)
            idle += settings.dashboard_events_seconds

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})  # fmt: skip


# ── Live order management ────────────────────────────────────────────────────


@router.get("/orders", response_model=OrderPage)
def list_orders(
    staff: StaffDep,
    db: SessionDep,
    status_filter: Annotated[OrderStatus | None, Query(alias="status")] = None,
    source: OrderSource | None = None,
    active_only: bool = False,
    q: Annotated[str | None, Query(max_length=100)] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
):
    """Orders from every channel, newest first."""
    items, total = service.list_orders(
        db, staff.restaurant_id, status=status_filter, source=source, active_only=active_only,
        search=q, limit=limit, offset=offset,
    )  # fmt: skip
    return OrderPage(items=items, total=total, limit=limit, offset=offset)


@router.get("/orders/{order_id}", response_model=OrderRead)
def get_order(order_id: uuid.UUID, staff: StaffDep, db: SessionDep):
    return service.get_scoped_order(db, staff.restaurant_id, order_id)


@router.post("/orders/{order_id}/status", response_model=OrderRead)
def update_status(order_id: uuid.UUID, body: OrderStatusUpdate, staff: StaffDep, db: SessionDep):
    """Advance preparation (confirmed → preparing → ready → completed) or
    cancel.  Same rules as the staff API: ``confirmed`` only via payment;
    cancelling settles stock and the demo payment."""
    service.get_scoped_order(db, staff.restaurant_id, order_id)
    order_service.update_order_status(db, order_id, body.status)
    return order_service.get_order(db, order_id)


@router.post("/orders/{order_id}/cancel", response_model=OrderRead)
def cancel(order_id: uuid.UUID, staff: StaffDep, db: SessionDep):
    service.get_scoped_order(db, staff.restaurant_id, order_id)
    order_service.cancel_order(db, order_id, CancelReason.STAFF)
    return order_service.get_order(db, order_id)


@router.post("/orders/{order_id}/mark-paid", response_model=OrderRead)
def mark_paid(order_id: uuid.UUID, staff: StaffDep, db: SessionDep):
    """DEMO PAYMENT MODE: record that a pending order was paid (e.g. cash at
    the counter).  The amount is the server's own payment amount; this goes
    through the same payment service, so stock moves from reserved to sold."""
    order = service.get_scoped_order(db, staff.restaurant_id, order_id)
    if order.payment is None:
        raise ConflictError("This order has no payment record.")
    if order.payment.status != PaymentStatus.PENDING.value:
        raise ConflictError(f"The payment is already '{order.payment.status}'.")
    payment_service.simulate(db, order.payment.id, DemoOutcome.SUCCEEDED, order.payment.amount_paise)
    logger.info("Dashboard marked order %s paid (demo)", order.order_number)
    return order_service.get_order(db, order_id)


@router.post("/orders/walk-in", status_code=status.HTTP_201_CREATED, response_model=OrderRead)
def record_walk_in(body: WalkInOrderCreate, staff: StaffDep, db: SessionDep):
    """Record a counter (walk-in) sale.  Prices come from the database, stock
    is reserved like any order, and the order is labelled WALK_IN_POS.  Mark
    it paid once the customer pays."""
    result = order_service.create_order(db, staff.restaurant_id, body, source=OrderSource.WALK_IN_POS)
    return result.order


# ── Menu & live inventory ────────────────────────────────────────────────────


@router.get("/menu", response_model=list[DashboardItem])
def list_menu(staff: StaffDep, db: SessionDep, include_inactive: bool = True):
    items, _ = menu_service.list_menu_items(
        db, staff.restaurant_id, limit=1000, offset=0, include_inactive=include_inactive
    )
    return [_item(i) for i in items]


@router.post("/menu", status_code=status.HTTP_201_CREATED, response_model=DashboardItem)
def create_item(body: NewMenuItem, staff: StaffDep, db: SessionDep):
    item = menu_service.create_menu_item(
        db, staff.restaurant_id, MenuItemCreate(**body.model_dump(exclude={"initial_stock"}))
    )
    if body.initial_stock is not None:
        item = inventory_service.set_stock(db, item.id, on_hand=body.initial_stock, note="initial stock")
    return _item(item)


@router.patch("/menu/{item_id}", response_model=DashboardItem)
def edit_item(item_id: uuid.UUID, body: MenuItemUpdate, staff: StaffDep, db: SessionDep):
    """Edit name, description, category, price or availability."""
    service.get_scoped_item(db, staff.restaurant_id, item_id)
    return _item(menu_service.update_menu_item(db, item_id, body))


@router.post("/menu/{item_id}/stock", response_model=DashboardItem)
def change_stock(item_id: uuid.UUID, body: StockChange, staff: StaffDep, db: SessionDep):
    """Add or remove physical stock with a reason (recorded in the ledger).
    Refused if it would take stock below what unpaid orders have reserved."""
    service.get_scoped_item(db, staff.restaurant_id, item_id)
    return _item(inventory_service.adjust_stock(db, item_id, change=body.change, reason=body.reason))


@router.post("/menu/{item_id}/deactivate", response_model=DashboardItem)
def deactivate_item(item_id: uuid.UUID, staff: StaffDep, db: SessionDep):
    """Retire a dish: it disappears from the menu, the voice agent and new
    orders, while past orders and the stock ledger are kept."""
    service.get_scoped_item(db, staff.restaurant_id, item_id)
    return _item(menu_service.set_menu_item_active(db, item_id, False))


@router.post("/menu/{item_id}/activate", response_model=DashboardItem)
def activate_item(item_id: uuid.UUID, staff: StaffDep, db: SessionDep):
    service.get_scoped_item(db, staff.restaurant_id, item_id)
    return _item(menu_service.set_menu_item_active(db, item_id, True))


@router.get("/menu/{item_id}/movements", response_model=list[MovementRead])
def item_movements(item_id: uuid.UUID, staff: StaffDep, db: SessionDep):
    service.get_scoped_item(db, staff.restaurant_id, item_id)
    return inventory_service.list_movements(db, item_id, limit=50, offset=0)

