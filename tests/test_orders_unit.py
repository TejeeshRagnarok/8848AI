"""Unit tests for orders: model metadata, status lifecycle, pricing,
request fingerprinting and request schemas.  No database connection.
"""

import itertools
import uuid
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import BigInteger, CheckConstraint, DateTime, Integer, UniqueConstraint, inspect

from app.database import Base
from app.models import Order, OrderItem, OrderStatus
from app.models.order import ALLOWED_TRANSITIONS, can_transition, order_number_seq
from app.schemas import OrderCreate, OrderStatusUpdate
from app.services.orders import price_lines, request_fingerprint

orders = Base.metadata.tables["orders"]
order_items = Base.metadata.tables["order_items"]


def _checks(table) -> dict[str, str]:
    return {c.name: str(c.sqltext) for c in table.constraints if isinstance(c, CheckConstraint)}


# ── Model metadata ───────────────────────────────────────────────────────────


class TestOrderMetadata:
    def test_tables_registered(self) -> None:
        assert {"orders", "order_items"} <= set(Base.metadata.tables)
        assert order_number_seq.metadata is Base.metadata
        assert (order_number_seq.name, order_number_seq.start) == ("order_number_seq", 100001)

    def test_required_and_optional_columns(self) -> None:
        optional = {"customer_name", "customer_phone", "notes", "reservation_expires_at", "cancel_reason"}
        for col in orders.columns:
            assert col.nullable is (col.name in optional), col.name

    def test_money_columns_are_bigint(self) -> None:
        assert isinstance(orders.c.subtotal_paise.type, BigInteger)
        assert isinstance(orders.c.total_paise.type, BigInteger)

    def test_check_constraints(self) -> None:
        checks = _checks(orders)
        assert checks["ck_orders_subtotal_paise_nonnegative"] == "subtotal_paise >= 0"
        assert checks["ck_orders_total_paise_nonnegative"] == "total_paise >= 0"
        for status in OrderStatus:
            assert f"'{status.value}'" in checks["ck_orders_status_valid"]

    def test_uniqueness(self) -> None:
        uniques = {
            c.name: [col.name for col in c.columns]
            for c in orders.constraints
            if isinstance(c, UniqueConstraint)
        }
        assert uniques == {
            "uq_orders_order_number": ["order_number"],
            "uq_orders_restaurant_id_idempotency_key": ["restaurant_id", "idempotency_key"],
        }

    def test_order_number_generated_by_database(self) -> None:
        assert "nextval('order_number_seq')" in str(orders.c.order_number.server_default.arg)

    def test_status_defaults_to_pending(self) -> None:
        assert orders.c.status.default.arg == "pending"
        assert orders.c.status.server_default.arg == "pending"

    def test_restaurant_fk_restricts_and_is_indexed(self) -> None:
        (fk,) = orders.c.restaurant_id.foreign_keys
        assert (fk.target_fullname, fk.ondelete) == ("restaurants.id", "RESTRICT")
        index_cols = {tuple(c.name for c in ix.columns) for ix in orders.indexes}
        assert ("restaurant_id", "created_at") in index_cols

    def test_timestamps_timezone_aware(self) -> None:
        for col in (orders.c.created_at, orders.c.updated_at, order_items.c.created_at):
            assert isinstance(col.type, DateTime) and col.type.timezone


class TestOrderItemMetadata:
    def test_required_and_optional_columns(self) -> None:
        for col in order_items.columns:
            assert col.nullable is (col.name == "menu_item_id"), col.name

    def test_no_updated_at_lines_are_immutable(self) -> None:
        assert "updated_at" not in order_items.c

    def test_column_types(self) -> None:
        assert isinstance(order_items.c.unit_price_paise.type, Integer)
        assert isinstance(order_items.c.quantity.type, Integer)
        assert isinstance(order_items.c.line_total_paise.type, BigInteger)

    def test_check_constraints(self) -> None:
        checks = _checks(order_items)
        assert checks["ck_order_items_quantity_positive"] == "quantity > 0"
        assert checks["ck_order_items_unit_price_paise_positive"] == "unit_price_paise > 0"
        assert (
            checks["ck_order_items_line_total_matches"]
            == "line_total_paise = unit_price_paise * quantity"
        )

    def test_foreign_key_policies(self) -> None:
        (order_fk,) = order_items.c.order_id.foreign_keys
        (menu_fk,) = order_items.c.menu_item_id.foreign_keys
        assert (order_fk.target_fullname, order_fk.ondelete) == ("orders.id", "CASCADE")
        # Deleting a menu item never deletes order history; the link is cleared.
        assert (menu_fk.target_fullname, menu_fk.ondelete) == ("menu_items.id", "SET NULL")

    def test_foreign_keys_indexed(self) -> None:
        index_cols = {tuple(c.name for c in ix.columns) for ix in order_items.indexes}
        assert {("order_id",), ("menu_item_id",)} <= index_cols


class TestRelationships:
    def test_order_items_relationship(self) -> None:
        rel = inspect(Order).relationships["items"]
        assert rel.mapper.class_ is OrderItem
        assert rel.back_populates == "order"
        assert [c.name for c in rel.order_by] == ["item_name", "id"]

    def test_item_order_relationship(self) -> None:
        rel = inspect(OrderItem).relationships["order"]
        assert rel.mapper.class_ is Order and rel.uselist is False

    def test_order_restaurant_relationship(self) -> None:
        assert inspect(Order).relationships["restaurant"].mapper.class_.__name__ == "Restaurant"


# ── Status lifecycle ─────────────────────────────────────────────────────────

VALID = {
    ("pending", "confirmed"), ("pending", "cancelled"),
    ("confirmed", "preparing"), ("confirmed", "cancelled"),
    ("preparing", "ready"), ("preparing", "cancelled"),
    ("ready", "completed"), ("ready", "cancelled"),
}  # fmt: skip


class TestStatusLifecycle:
    @pytest.mark.parametrize(
        ("current", "new"), list(itertools.product([s.value for s in OrderStatus], repeat=2))
    )
    def test_transition_table(self, current: str, new: str) -> None:
        assert can_transition(current, new) is ((current, new) in VALID)

    def test_terminal_states(self) -> None:
        assert ALLOWED_TRANSITIONS[OrderStatus.COMPLETED] == frozenset()
        assert ALLOWED_TRANSITIONS[OrderStatus.CANCELLED] == frozenset()

    def test_every_status_has_a_rule(self) -> None:
        assert set(ALLOWED_TRANSITIONS) == set(OrderStatus)

    def test_status_update_schema_rejects_unknown_status(self) -> None:
        with pytest.raises(ValidationError):
            OrderStatusUpdate(status="paid")


# ── Pricing and fingerprint ──────────────────────────────────────────────────


def _line(item_id: uuid.UUID, quantity: int):
    return SimpleNamespace(menu_item_id=item_id, quantity=quantity)


class TestPricing:
    def test_line_totals_use_database_prices(self) -> None:
        a, b = uuid.uuid4(), uuid.uuid4()
        menu = {
            a: SimpleNamespace(name="Masala Dosa", price_paise=12050),
            b: SimpleNamespace(name="Filter Coffee", price_paise=4000),
        }
        priced = price_lines(menu, [_line(a, 3), _line(b, 2)])
        assert [(p.item_name, p.unit_price_paise, p.quantity, p.line_total_paise) for p in priced] == [
            ("Masala Dosa", 12050, 3, 36150),
            ("Filter Coffee", 4000, 2, 8000),
        ]
        assert sum(p.line_total_paise for p in priced) == 44150

    def test_totals_are_exact_integers(self) -> None:
        a = uuid.uuid4()
        (line,) = price_lines({a: SimpleNamespace(name="X", price_paise=2_147_483_647)}, [_line(a, 100)])
        assert line.line_total_paise == 214_748_364_700
        assert isinstance(line.line_total_paise, int)


def _order(**overrides) -> OrderCreate:
    data = {
        "customer_name": "Asha",
        "customer_phone": "+91 98765 43210",
        "items": [
            {"menu_item_id": "00000000-0000-0000-0000-000000000001", "quantity": 1},
            {"menu_item_id": "00000000-0000-0000-0000-000000000002", "quantity": 2},
        ],
        "idempotency_key": "call-123",
    }
    return OrderCreate(**(data | overrides))


class TestRequestFingerprint:
    def test_stable_and_order_insensitive(self) -> None:
        reordered = _order(items=list(reversed(_order().model_dump()["items"])))
        assert request_fingerprint(_order()) == request_fingerprint(reordered)

    def test_ignores_idempotency_key(self) -> None:
        assert request_fingerprint(_order()) == request_fingerprint(_order(idempotency_key="other"))

    @pytest.mark.parametrize(
        "change",
        [
            {"customer_name": "Ravi"},
            {"customer_phone": "+91 11111 11111"},
            {"notes": "extra spicy"},
            {"items": [{"menu_item_id": "00000000-0000-0000-0000-000000000001", "quantity": 2}]},
        ],
    )
    def test_detects_different_requests(self, change: dict) -> None:
        assert request_fingerprint(_order()) != request_fingerprint(_order(**change))

    def test_is_sha256_hex(self) -> None:
        fp = request_fingerprint(_order())
        assert len(fp) == 64 and int(fp, 16) >= 0


# ── Request schema ───────────────────────────────────────────────────────────


class TestOrderCreateSchema:
    def test_valid_minimal(self) -> None:
        order = _order(customer_name=None, idempotency_key=None)
        assert order.notes is None and order.idempotency_key is None

    @pytest.mark.parametrize("quantity", [0, -1, -100, 101, 1.5, 2.0, "2", True])
    def test_quantity_must_be_positive_integer(self, quantity) -> None:
        with pytest.raises(ValidationError):
            _order(items=[{"menu_item_id": str(uuid.uuid4()), "quantity": quantity}])

    @pytest.mark.parametrize(
        "field", ["unit_price_paise", "line_total_paise", "price_paise", "item_name"]
    )
    def test_client_prices_rejected_on_lines(self, field: str) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _order(items=[{"menu_item_id": str(uuid.uuid4()), "quantity": 1, field: 1}])

    @pytest.mark.parametrize(
        "field", ["subtotal_paise", "total_paise", "status", "order_number", "id", "restaurant_id"]
    )
    def test_client_totals_and_server_fields_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _order(**{field: 1})

    def test_items_required_and_bounded(self) -> None:
        with pytest.raises(ValidationError):
            _order(items=[])
        too_many = [{"menu_item_id": str(uuid.uuid4()), "quantity": 1} for _ in range(51)]
        with pytest.raises(ValidationError):
            _order(items=too_many)

    def test_duplicate_menu_items_rejected(self) -> None:
        same = str(uuid.uuid4())
        with pytest.raises(ValidationError, match="only once"):
            _order(items=[{"menu_item_id": same, "quantity": 1}, {"menu_item_id": same, "quantity": 2}])

    @pytest.mark.parametrize("phone", [None, "", "abc", "123"])
    def test_customer_phone_required_and_valid(self, phone) -> None:
        with pytest.raises(ValidationError):
            _order(customer_phone=phone)

    @pytest.mark.parametrize("key", ["", "has space", "x" * 101, "semi;colon"])
    def test_idempotency_key_format(self, key: str) -> None:
        with pytest.raises(ValidationError):
            _order(idempotency_key=key)

    def test_notes_length(self) -> None:
        with pytest.raises(ValidationError):
            _order(notes="x" * 501)
