"""Live tests for the Restaurant and MenuItem models against real PostgreSQL.

Marked ``live_db``; skipped automatically when the database is unreachable or
not migrated to head.  Fixtures (in ``conftest.py``) wrap every test in an
outer transaction that is always rolled back, so nothing is ever written to
the real database.

Run only these:   pytest -m live_db
Skip them:        pytest -m "not live_db"
"""

import uuid
from datetime import datetime

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Connection, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import Base
from app.models import MenuItem, Restaurant

pytestmark = pytest.mark.live_db


def _slug() -> str:
    return f"test-{uuid.uuid4().hex[:12]}"


def _restaurant(session: Session, **kwargs) -> Restaurant:
    restaurant = Restaurant(name="Test Kitchen", slug=_slug(), **kwargs)
    session.add(restaurant)
    session.flush()
    return restaurant


def _assert_violates(session: Session, constraint: str) -> None:
    with pytest.raises(IntegrityError) as exc_info:
        session.flush()
    assert constraint in str(exc_info.value.orig)
    session.rollback()


class TestSchema:
    def test_database_schema_matches_models(self, connection: Connection) -> None:
        diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)
        assert diff == []


class TestRestaurantPersistence:
    def test_defaults_applied_on_insert(self, session: Session) -> None:
        restaurant = _restaurant(session)
        assert isinstance(restaurant.id, uuid.UUID)
        assert restaurant.timezone == "Asia/Kolkata"
        assert restaurant.currency == "INR"
        assert restaurant.is_active is True
        assert restaurant.phone is None

    def test_server_defaults_apply_to_raw_sql_inserts(self, connection: Connection) -> None:
        row = connection.execute(
            text(
                "INSERT INTO restaurants (id, name, slug) VALUES (:id, 'Raw', :slug) "
                "RETURNING timezone, currency, is_active, created_at, updated_at"
            ),
            {"id": uuid.uuid4(), "slug": _slug()},
        ).one()
        assert row.timezone == "Asia/Kolkata"
        assert row.currency == "INR"
        assert row.is_active is True
        assert row.created_at.tzinfo is not None
        assert row.updated_at == row.created_at

    def test_slug_must_be_unique(self, session: Session) -> None:
        existing = _restaurant(session)
        session.add(Restaurant(name="Copy", slug=existing.slug))
        _assert_violates(session, "uq_restaurants_slug")

    def test_name_is_required(self, session: Session) -> None:
        session.add(Restaurant(slug=_slug()))
        _assert_violates(session, "name")

    @pytest.mark.parametrize("slug", ["Has-Caps", "two--hyphens", "-leading", "spa ce", ""])
    def test_slug_must_be_url_safe(self, session: Session, slug: str) -> None:
        session.add(Restaurant(name="Bad Slug", slug=slug))
        _assert_violates(session, "ck_restaurants_slug_format")

    @pytest.mark.parametrize("currency", ["inr", "IN", "1NR"])
    def test_currency_must_be_iso_code(self, session: Session, currency: str) -> None:
        session.add(Restaurant(name="Bad Currency", slug=_slug(), currency=currency))
        _assert_violates(session, "ck_restaurants_currency_format")


class TestMenuItemPersistence:
    def test_defaults_applied_on_insert(self, session: Session) -> None:
        restaurant = _restaurant(session)
        item = MenuItem(restaurant=restaurant, name="Masala Dosa", price_paise=12000)
        session.add(item)
        session.flush()
        assert isinstance(item.id, uuid.UUID)
        assert item.restaurant_id == restaurant.id
        assert item.is_available is True
        assert item.description is None and item.category is None

    def test_foreign_key_requires_existing_restaurant(self, session: Session) -> None:
        session.add(MenuItem(restaurant_id=uuid.uuid4(), name="Orphan", price_paise=100))
        _assert_violates(session, "fk_menu_items_restaurant_id_restaurants")

    @pytest.mark.parametrize("price", [0, -1])
    def test_price_must_be_positive(self, session: Session, price: int) -> None:
        restaurant = _restaurant(session)
        session.add(MenuItem(restaurant=restaurant, name="Free Lunch", price_paise=price))
        _assert_violates(session, "ck_menu_items_price_paise_positive")


class TestRelationships:
    def test_restaurant_loads_its_menu_items(self, session: Session) -> None:
        restaurant = _restaurant(session)
        other = _restaurant(session)
        restaurant.menu_items.extend(
            [MenuItem(name="Idli", price_paise=6000), MenuItem(name="Vada", price_paise=5000)]
        )
        other.menu_items.append(MenuItem(name="Elsewhere", price_paise=100))
        session.flush()
        session.expire_all()

        loaded = session.get(Restaurant, restaurant.id)
        assert sorted(i.name for i in loaded.menu_items) == ["Idli", "Vada"]
        assert all(i.restaurant is loaded for i in loaded.menu_items)

        by_restaurant = session.scalars(
            select(MenuItem.name).where(MenuItem.restaurant_id == other.id)
        ).all()
        assert by_restaurant == ["Elsewhere"]

    def test_deleting_restaurant_with_menu_items_is_restricted(self, session: Session) -> None:
        restaurant = _restaurant(session)
        restaurant.menu_items.append(MenuItem(name="Idli", price_paise=6000))
        session.flush()
        session.delete(restaurant)
        _assert_violates(session, "fk_menu_items_restaurant_id_restaurants")

    def test_deleting_restaurant_without_menu_items_succeeds(self, session: Session) -> None:
        restaurant = _restaurant(session)
        session.delete(restaurant)
        session.flush()
        assert session.get(Restaurant, restaurant.id) is None


class TestTimestamps:
    def _refreshed(self, session: Session, obj) -> tuple[datetime, datetime]:
        session.refresh(obj)
        return obj.created_at, obj.updated_at

    def test_timestamps_set_and_timezone_aware(self, session: Session) -> None:
        created, updated = self._refreshed(session, _restaurant(session))
        assert created.tzinfo is not None and updated.tzinfo is not None
        assert created == updated

    @pytest.mark.parametrize("model", ["restaurant", "menu_item"])
    def test_updated_at_advances_on_update(self, session: Session, model: str) -> None:
        obj = _restaurant(session)
        if model == "menu_item":
            obj = MenuItem(restaurant=obj, name="Dosa", price_paise=100)
            session.add(obj)
            session.flush()
        created_before, updated_before = self._refreshed(session, obj)

        obj.name = "Renamed"
        session.flush()
        created_after, updated_after = self._refreshed(session, obj)

        assert created_after == created_before
        assert updated_after > updated_before
