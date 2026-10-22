"""Restaurant CRUD endpoints (writes need X-Staff-Key)."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.security import require_staff
from app.schemas import Page, RestaurantCreate, RestaurantRead, RestaurantUpdate
from app.schemas.common import MAX_PAGE_SIZE
from app.services import restaurants as service

router = APIRouter(prefix="/api/v1/restaurants", tags=["restaurants"])

SessionDep = Annotated[Session, Depends(get_db)]
StaffOnly = [Depends(require_staff)]


@router.post("", status_code=status.HTTP_201_CREATED, response_model=RestaurantRead, dependencies=StaffOnly)
def create_restaurant(body: RestaurantCreate, db: SessionDep):
    """Create a restaurant.  409 if the slug is already taken."""
    return service.create_restaurant(db, body)


@router.get("", response_model=Page[RestaurantRead])
def list_restaurants(
    db: SessionDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    include_inactive: bool = False,
):
    """List restaurants, oldest first.  Inactive ones only with ``include_inactive=true``."""
    items, total = service.list_restaurants(
        db, limit=limit, offset=offset, include_inactive=include_inactive
    )
    return Page[RestaurantRead](items=items, total=total, limit=limit, offset=offset)


@router.get("/{restaurant_id}", response_model=RestaurantRead)
def get_restaurant(restaurant_id: uuid.UUID, db: SessionDep):
    """Fetch one restaurant (active or not — check ``is_active``)."""
    return service.get_restaurant(db, restaurant_id)


@router.patch("/{restaurant_id}", response_model=RestaurantRead, dependencies=StaffOnly)
def update_restaurant(restaurant_id: uuid.UUID, body: RestaurantUpdate, db: SessionDep):
    """Partially update a restaurant.  The ID cannot be changed."""
    return service.update_restaurant(db, restaurant_id, body)


@router.delete("/{restaurant_id}", response_model=RestaurantRead, dependencies=StaffOnly)
def deactivate_restaurant(restaurant_id: uuid.UUID, db: SessionDep):
    """Deactivate (soft-delete) a restaurant; the row and its menu are kept."""
    return service.deactivate_restaurant(db, restaurant_id)
