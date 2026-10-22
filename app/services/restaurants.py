"""Restaurant persistence operations."""

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Restaurant
from app.schemas import RestaurantCreate, RestaurantUpdate
from app.services.common import NotFoundError, commit


def get_restaurant(session: Session, restaurant_id: uuid.UUID, *, lock: bool = False) -> Restaurant:
    """Return the restaurant or raise ``NotFoundError``.

    ``lock=True`` takes a shared row lock (``FOR SHARE``) until the transaction
    ends, so the restaurant cannot be deactivated concurrently, and reloads
    the row so a stale copy in the session is never trusted.
    """
    restaurant = session.get(
        Restaurant,
        restaurant_id,
        with_for_update={"read": True} if lock else None,
        populate_existing=lock,
    )
    if restaurant is None:
        raise NotFoundError("Restaurant not found.")
    return restaurant


def list_restaurants(
    session: Session, *, limit: int, offset: int, include_inactive: bool = False
) -> tuple[list[Restaurant], int]:
    """Return one page of restaurants (active only by default) and the total count."""
    conditions = [] if include_inactive else [Restaurant.is_active.is_(True)]
    total = session.scalar(select(func.count()).select_from(Restaurant).where(*conditions))
    items = session.scalars(
        select(Restaurant)
        .where(*conditions)
        .order_by(Restaurant.created_at, Restaurant.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return list(items), total or 0


def create_restaurant(session: Session, data: RestaurantCreate) -> Restaurant:
    restaurant = Restaurant(**data.model_dump())
    session.add(restaurant)
    commit(session)
    return restaurant


def update_restaurant(
    session: Session, restaurant_id: uuid.UUID, data: RestaurantUpdate
) -> Restaurant:
    restaurant = get_restaurant(session, restaurant_id)
    for field, value in data.changes().items():
        setattr(restaurant, field, value)
    commit(session)
    return restaurant


def deactivate_restaurant(session: Session, restaurant_id: uuid.UUID) -> Restaurant:
    """Soft-delete: mark inactive, keeping the row and its menu.  Idempotent."""
    restaurant = get_restaurant(session, restaurant_id)
    restaurant.is_active = False
    commit(session)
    return restaurant
