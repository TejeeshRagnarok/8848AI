"""Menu item CRUD endpoints (writes need X-Staff-Key)."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.security import require_staff
from app.schemas import MenuItemCreate, MenuItemRead, MenuItemUpdate, Page
from app.schemas.common import MAX_PAGE_SIZE
from app.services import menu_items as service

router = APIRouter(prefix="/api/v1", tags=["menu items"])

SessionDep = Annotated[Session, Depends(get_db)]
StaffOnly = [Depends(require_staff)]


@router.post(
    "/restaurants/{restaurant_id}/menu-items",
    status_code=status.HTTP_201_CREATED,
    response_model=MenuItemRead,
    dependencies=StaffOnly,
)
def create_menu_item(restaurant_id: uuid.UUID, body: MenuItemCreate, db: SessionDep):
    """Add an item to a restaurant's menu.  409 if the restaurant is inactive."""
    return service.create_menu_item(db, restaurant_id, body)


@router.get("/restaurants/{restaurant_id}/menu-items", response_model=Page[MenuItemRead])
def list_menu_items(
    restaurant_id: uuid.UUID,
    db: SessionDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    is_available: bool | None = None,
    in_stock: bool | None = None,
):
    """List a restaurant's menu items by name, optionally filtered by
    availability switch and stock (untracked items count as in stock).
    ``orderable`` and ``available_quantity`` show live stock."""
    items, total = service.list_menu_items(
        db, restaurant_id, limit=limit, offset=offset, is_available=is_available,
        in_stock=in_stock,
    )  # fmt: skip
    return Page[MenuItemRead](items=items, total=total, limit=limit, offset=offset)


@router.get("/menu-items/{item_id}", response_model=MenuItemRead)
def get_menu_item(item_id: uuid.UUID, db: SessionDep):
    return service.get_menu_item(db, item_id)


@router.patch("/menu-items/{item_id}", response_model=MenuItemRead, dependencies=StaffOnly)
def update_menu_item(item_id: uuid.UUID, body: MenuItemUpdate, db: SessionDep):
    """Partially update an item, e.g. its price or availability."""
    return service.update_menu_item(db, item_id, body)


@router.delete("/menu-items/{item_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=StaffOnly)
def delete_menu_item(item_id: uuid.UUID, db: SessionDep) -> Response:
    """Permanently delete a menu item."""
    service.delete_menu_item(db, item_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
