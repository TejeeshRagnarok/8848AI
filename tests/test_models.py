"""Unit tests for the Restaurant and MenuItem models.

These inspect SQLAlchemy metadata and mappers only — no database connection.
Behaviour against real PostgreSQL is covered in ``test_models_live.py``.
"""

import uuid

from sqlalchemy import CheckConstraint, DateTime, Uuid, inspect

from app.database import Base
from app.models import MenuItem, Restaurant

restaurants = Base.metadata.tables["restaurants"]
menu_items = Base.metadata.tables["menu_items"]


def _check_sql(table, name: str) -> str:
    """Return the SQL text of the named check constraint on ``table``."""
    for c in table.constraints:
        if isinstance(c, CheckConstraint) and c.name == name:
            return str(c.sqltext)
    raise AssertionError(f"check constraint {name!r} not found on {table.name}")


class TestRegistration:
    def test_both_models_registered_in_metadata(self) -> None:
        assert {"restaurants", "menu_items"} <= set(Base.metadata.tables)

    def test_models_share_the_single_app_base(self) -> None:
        assert Restaurant.metadata is Base.metadata
        assert MenuItem.metadata is Base.metadata


class TestRestaurantColumns:
    def test_uuid_primary_key_generated_by_application(self) -> None:
        col = restaurants.c.id
        assert col.primary_key
        assert isinstance(col.type, Uuid)
        assert col.default.arg.__name__ == "uuid4"

    def test_required_and_optional_columns(self) -> None:
        required = {"id", "name", "slug", "timezone", "currency", "is_active",
                    "created_at", "updated_at"}
        for name in required:
            assert restaurants.c[name].nullable is False, name
        assert restaurants.c.phone.nullable is True

    def test_column_lengths(self) -> None:
        assert restaurants.c.name.type.length == 200
        assert restaurants.c.slug.type.length == 100
        assert restaurants.c.phone.type.length == 20
        assert restaurants.c.timezone.type.length == 64
        assert restaurants.c.currency.type.length == 3

    def test_defaults_are_set_in_python_and_database(self) -> None:
        for name, value in [("timezone", "Asia/Kolkata"), ("currency", "INR")]:
            assert restaurants.c[name].default.arg == value
            assert restaurants.c[name].server_default.arg == value
        assert restaurants.c.is_active.default.arg is True
        assert str(restaurants.c.is_active.server_default.arg) == "true"

    def test_slug_is_unique(self) -> None:
        uniques = [c for c in restaurants.constraints if c.__class__.__name__ == "UniqueConstraint"]
        assert [[col.name for col in u.columns] for u in uniques] == [["slug"]]
        assert uniques[0].name == "uq_restaurants_slug"

    def test_format_check_constraints(self) -> None:
        assert "slug ~" in _check_sql(restaurants, "ck_restaurants_slug_format")
        assert "currency ~" in _check_sql(restaurants, "ck_restaurants_currency_format")


class TestMenuItemColumns:
    def test_uuid_primary_key_generated_by_application(self) -> None:
        assert menu_items.c.id.primary_key
        assert isinstance(menu_items.c.id.type, Uuid)
        assert menu_items.c.id.default.arg.__name__ == "uuid4"

    def test_required_and_optional_columns(self) -> None:
        required = {"id", "restaurant_id", "name", "price_paise", "is_available",
                    "created_at", "updated_at"}
        for name in required:
            assert menu_items.c[name].nullable is False, name
        assert menu_items.c.description.nullable is True
        assert menu_items.c.category.nullable is True

    def test_is_available_defaults_true(self) -> None:
        assert menu_items.c.is_available.default.arg is True
        assert str(menu_items.c.is_available.server_default.arg) == "true"

    def test_foreign_key_restricts_restaurant_deletion(self) -> None:
        (fk,) = menu_items.c.restaurant_id.foreign_keys
        assert fk.target_fullname == "restaurants.id"
        assert fk.ondelete == "RESTRICT"

    def test_restaurant_id_is_indexed(self) -> None:
        indexed = {tuple(c.name for c in ix.columns) for ix in menu_items.indexes}
        assert ("restaurant_id",) in indexed

    def test_positive_price_check_constraint(self) -> None:
        assert _check_sql(menu_items, "ck_menu_items_price_paise_positive") == "price_paise > 0"


class TestTimestamps:
    def test_timestamps_are_timezone_aware_with_server_defaults(self) -> None:
        for table in (restaurants, menu_items):
            for name in ("created_at", "updated_at"):
                col = table.c[name]
                assert isinstance(col.type, DateTime) and col.type.timezone, name
                assert col.server_default is not None, name

    def test_updated_at_refreshes_on_update(self) -> None:
        for table in (restaurants, menu_items):
            assert table.c.updated_at.onupdate is not None
            assert table.c.created_at.onupdate is None


class TestRelationships:
    def test_restaurant_menu_items_relationship(self) -> None:
        rel = inspect(Restaurant).relationships["menu_items"]
        assert rel.mapper.class_ is MenuItem
        assert rel.back_populates == "restaurant"
        assert rel.uselist is True
        # The ORM must not null out or delete children; the DB restricts instead.
        assert rel.passive_deletes == "all"
        assert "delete" not in rel.cascade

    def test_menu_item_restaurant_relationship(self) -> None:
        rel = inspect(MenuItem).relationships["restaurant"]
        assert rel.mapper.class_ is Restaurant
        assert rel.back_populates == "menu_items"
        assert rel.uselist is False

    def test_in_memory_back_population(self) -> None:
        restaurant = Restaurant(name="Spice Garden", slug="spice-garden")
        item = MenuItem(name="Masala Dosa", price_paise=12000)
        restaurant.menu_items.append(item)
        assert item.restaurant is restaurant

    def test_ids_are_uuids_when_set(self) -> None:
        rid = uuid.uuid4()
        assert Restaurant(id=rid, name="x", slug="x").id == rid
