"""Unit tests for request/response schemas — pure Pydantic, no database."""

import pytest
from pydantic import ValidationError

from app.schemas import MenuItemCreate, MenuItemUpdate, RestaurantCreate, RestaurantUpdate


class TestRestaurantCreate:
    def test_minimal_valid_payload_gets_defaults(self) -> None:
        r = RestaurantCreate(name="  Spice Garden  ", slug="spice-garden")
        assert r.name == "Spice Garden"  # trimmed
        assert r.timezone == "Asia/Kolkata"
        assert r.currency == "INR"
        assert r.phone is None

    @pytest.mark.parametrize("field", ["name", "slug"])
    def test_required_fields(self, field: str) -> None:
        data = {"name": "Spice Garden", "slug": "spice-garden"}
        del data[field]
        with pytest.raises(ValidationError):
            RestaurantCreate(**data)

    @pytest.mark.parametrize(
        "changes",
        [
            {"name": ""},
            {"name": "   "},
            {"name": "x" * 201},
            {"slug": "Spice-Garden"},
            {"slug": "spice--garden"},
            {"slug": "x" * 101},
            {"currency": "inr"},
            {"currency": "INRS"},
            {"timezone": "Mars/Olympus"},
            {"timezone": "../etc/passwd"},
            {"phone": "call me"},
            {"phone": "123"},
        ],
    )
    def test_invalid_values_rejected(self, changes: dict) -> None:
        with pytest.raises(ValidationError):
            RestaurantCreate(**({"name": "Spice Garden", "slug": "spice-garden"} | changes))

    @pytest.mark.parametrize("field", ["id", "is_active", "created_at", "updated_at"])
    def test_server_managed_fields_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RestaurantCreate(name="Spice Garden", slug="spice-garden", **{field: "x"})

    def test_valid_optional_fields(self) -> None:
        r = RestaurantCreate(
            name="Spice Garden",
            slug="spice-garden-2",
            phone="+91 98765-43210",
            timezone="Europe/London",
            currency="GBP",
        )
        assert r.timezone == "Europe/London"


class TestRestaurantUpdate:
    def test_only_sent_fields_are_changes(self) -> None:
        assert RestaurantUpdate(name="New Name").changes() == {"name": "New Name"}
        assert RestaurantUpdate().changes() == {}

    def test_nullable_field_can_be_cleared(self) -> None:
        assert RestaurantUpdate(phone=None).changes() == {"phone": None}

    @pytest.mark.parametrize("field", ["name", "slug", "timezone", "currency", "is_active"])
    def test_required_columns_cannot_be_nulled(self, field: str) -> None:
        with pytest.raises(ValidationError, match="cannot be null"):
            RestaurantUpdate(**{field: None})

    def test_id_cannot_be_changed(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RestaurantUpdate(id="00000000-0000-0000-0000-000000000000")

    def test_non_nullable_is_not_a_request_field(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RestaurantUpdate(non_nullable=[])


class TestMenuItemSchemas:
    def test_valid_create(self) -> None:
        item = MenuItemCreate(name="Masala Dosa", price_paise=12000)
        assert item.is_available is True
        assert item.description is None and item.category is None

    @pytest.mark.parametrize("price", [0, -1, -12000])
    def test_non_positive_price_rejected(self, price: int) -> None:
        with pytest.raises(ValidationError):
            MenuItemCreate(name="Masala Dosa", price_paise=price)

    @pytest.mark.parametrize("price", [120.5, 120.0, "12000", True, 2_147_483_648])
    def test_price_must_be_integer_paise_in_range(self, price) -> None:
        with pytest.raises(ValidationError):
            MenuItemCreate(name="Masala Dosa", price_paise=price)

    @pytest.mark.parametrize("field", ["id", "restaurant_id", "created_at"])
    def test_server_managed_fields_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MenuItemCreate(name="Dosa", price_paise=100, **{field: "x"})

    def test_update_is_partial(self) -> None:
        assert MenuItemUpdate(price_paise=15000).changes() == {"price_paise": 15000}
        assert MenuItemUpdate(category=None).changes() == {"category": None}

    @pytest.mark.parametrize("field", ["name", "price_paise", "is_available"])
    def test_update_required_columns_cannot_be_nulled(self, field: str) -> None:
        with pytest.raises(ValidationError, match="cannot be null"):
            MenuItemUpdate(**{field: None})

    @pytest.mark.parametrize("price", [0, -5])
    def test_update_rejects_non_positive_price(self, price: int) -> None:
        with pytest.raises(ValidationError):
            MenuItemUpdate(price_paise=price)

    def test_update_cannot_move_item_to_another_restaurant(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MenuItemUpdate(restaurant_id="00000000-0000-0000-0000-000000000000")
