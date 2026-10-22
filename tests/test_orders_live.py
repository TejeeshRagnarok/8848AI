"""Live order API tests: HTTP requests through FastAPI into real PostgreSQL.

Marked ``live_db``.  All requests share one database session inside a
transaction that is rolled back after each test (see ``conftest.py``), so no
restaurants, menu items or orders are left behind.
"""

import logging
import uuid

import pytest
from db_helpers import count_test_orders
from fastapi.testclient import TestClient
from sqlalchemy import Connection, func, select

from app.models import Order, OrderItem

pytestmark = pytest.mark.live_db

API = "/api/v1"
PHONE = "+91 98765 43210"


def _restaurant(client: TestClient) -> str:
    response = client.post(
        f"{API}/restaurants", json={"name": "Test Kitchen", "slug": f"test-{uuid.uuid4().hex[:12]}"}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _item(client: TestClient, restaurant_id: str, name: str, price: int, **extra) -> str:
    response = client.post(
        f"{API}/restaurants/{restaurant_id}/menu-items",
        json={"name": name, "price_paise": price} | extra,
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _order_body(lines: list[tuple[str, int]], **extra) -> dict:
    return {
        "customer_name": "Asha",
        "customer_phone": PHONE,
        "items": [{"menu_item_id": i, "quantity": q} for i, q in lines],
    } | extra


def _place(client: TestClient, restaurant_id: str, body: dict):
    return client.post(f"{API}/restaurants/{restaurant_id}/orders", json=body)


def _pay(client: TestClient, order_id) -> object:
    """Confirm an order the only way possible: its demo payment succeeds."""
    order = client.get(f"{API}/orders/{order_id}")
    if order.status_code != 200:
        return order
    payment = order.json()["payment"]
    return client.post(
        f"{API}/payments/{payment['id']}/simulate",
        json={"outcome": "succeeded", "amount_paise": payment["amount_paise"]},
    )


def _order_count(connection: Connection) -> int:
    return count_test_orders(connection)


@pytest.fixture
def menu(live_client: TestClient) -> dict:
    rid = _restaurant(live_client)
    return {
        "restaurant": rid,
        "dosa": _item(live_client, rid, "Masala Dosa", 12050),
        "coffee": _item(live_client, rid, "Filter Coffee", 4000),
        "vada": _item(live_client, rid, "Medu Vada", 6000),
    }


class TestCreateOrder:
    def test_valid_order_with_server_side_pricing(self, live_client, menu) -> None:
        response = _place(
            live_client, menu["restaurant"],
            _order_body([(menu["dosa"], 3), (menu["coffee"], 2), (menu["vada"], 1)],
                        notes="Less spicy", idempotency_key="call-1"),
        )  # fmt: skip
        assert response.status_code == 201, response.text
        order = response.json()
        assert order["status"] == "pending"
        assert order["order_number"].startswith("ORD-")
        assert order["customer_phone"] == PHONE
        assert order["idempotency_key"] == "call-1"
        lines = {i["item_name"]: i for i in order["items"]}
        assert {n: (l["unit_price_paise"], l["quantity"], l["line_total_paise"])
                for n, l in lines.items()} == {
            "Masala Dosa": (12050, 3, 36150),
            "Filter Coffee": (4000, 2, 8000),
            "Medu Vada": (6000, 1, 6000),
        }  # fmt: skip
        assert order["subtotal_paise"] == 50150
        assert order["total_paise"] == order["subtotal_paise"]

        assert order["inventory_state"] == "none"  # no stock-tracked items
        assert order["reservation_expires_at"] is not None
        assert order["payment"]["status"] == "pending"
        assert order["payment"]["amount_paise"] == order["total_paise"]
        assert order["payment"]["mode"] == "demo" and order["payment"]["simulated"] is True
        assert order["customer_token"]

        fetched = live_client.get(f"{API}/orders/{order['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == {k: v for k, v in order.items() if k != "customer_token"}

    def test_order_numbers_are_unique(self, live_client, menu) -> None:
        numbers = {
            _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)])).json()["order_number"]
            for _ in range(3)
        }
        assert len(numbers) == 3

    def test_server_generates_idempotency_key_when_omitted(self, live_client, menu) -> None:
        order = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)])).json()
        assert order["idempotency_key"].startswith("auto-")

    def test_missing_restaurant_returns_404(self, live_client, menu) -> None:
        response = _place(live_client, str(uuid.uuid4()), _order_body([(menu["dosa"], 1)]))
        assert response.status_code == 404

    def test_inactive_restaurant_returns_409(self, live_client, menu, connection) -> None:
        live_client.delete(f"{API}/restaurants/{menu['restaurant']}")
        response = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)]))
        assert response.status_code == 409
        assert "inactive" in response.json()["detail"]
        assert _order_count(connection) == 0

    def test_missing_menu_item_returns_422(self, live_client, menu, connection) -> None:
        missing = str(uuid.uuid4())
        response = _place(
            live_client, menu["restaurant"], _order_body([(menu["dosa"], 1), (missing, 1)])
        )
        assert response.status_code == 422
        (error,) = response.json()["detail"]
        assert error["type"] == "menu_item_not_found"
        assert error["loc"] == ["body", "items", 1, "menu_item_id"]
        assert _order_count(connection) == 0

    def test_item_from_another_restaurant_returns_422(self, live_client, menu, connection) -> None:
        other = _restaurant(live_client)
        foreign = _item(live_client, other, "Foreign Dish", 999)
        response = _place(live_client, menu["restaurant"], _order_body([(foreign, 1)]))
        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] == "menu_item_not_found"
        assert _order_count(connection) == 0

    def test_unavailable_item_returns_422(self, live_client, menu, connection) -> None:
        live_client.patch(f"{API}/menu-items/{menu['vada']}", json={"is_available": False})
        response = _place(
            live_client, menu["restaurant"], _order_body([(menu["dosa"], 1), (menu["vada"], 2)])
        )
        assert response.status_code == 422
        (error,) = response.json()["detail"]
        assert error["type"] == "menu_item_unavailable"
        assert "Medu Vada" in error["msg"]
        assert _order_count(connection) == 0

    @pytest.mark.parametrize("quantity", [0, -1])
    def test_non_positive_quantity_returns_422(self, live_client, menu, connection, quantity) -> None:
        response = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], quantity)]))
        assert response.status_code == 422
        assert _order_count(connection) == 0

    def test_client_supplied_prices_rejected(self, live_client, menu) -> None:
        body = _order_body([(menu["dosa"], 1)])
        body["items"][0]["unit_price_paise"] = 1
        assert _place(live_client, menu["restaurant"], body).status_code == 422
        assert _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)], total_paise=1)).status_code == 422


class TestIdempotency:
    def test_retry_with_same_request_returns_existing_order(self, live_client, menu, connection) -> None:
        body = _order_body([(menu["dosa"], 2), (menu["coffee"], 1)], idempotency_key="call-42")
        first = _place(live_client, menu["restaurant"], body)
        # Same lines in a different order still count as the same request.
        retry_body = body | {"items": list(reversed(body["items"]))}
        retry = _place(live_client, menu["restaurant"], retry_body)

        assert first.status_code == 201
        assert retry.status_code == 200
        assert retry.headers["Idempotent-Replayed"] == "true"
        assert retry.json() == first.json()
        assert _order_count(connection) == 1

    def test_same_key_with_different_request_returns_409(self, live_client, menu, connection) -> None:
        _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)], idempotency_key="k1"))
        response = _place(
            live_client, menu["restaurant"], _order_body([(menu["dosa"], 2)], idempotency_key="k1")
        )
        assert response.status_code == 409
        assert "different order request" in response.json()["detail"]
        assert _order_count(connection) == 1

    def test_keys_are_scoped_per_restaurant(self, live_client, menu, connection) -> None:
        other = _restaurant(live_client)
        other_item = _item(live_client, other, "Idli", 3000)
        a = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)], idempotency_key="shared"))
        b = _place(live_client, other, _order_body([(other_item, 1)], idempotency_key="shared"))
        assert (a.status_code, b.status_code) == (201, 201)
        assert a.json()["id"] != b.json()["id"]

    def test_replay_returns_original_even_after_price_change(self, live_client, menu) -> None:
        body = _order_body([(menu["dosa"], 1)], idempotency_key="call-7")
        first = _place(live_client, menu["restaurant"], body).json()
        live_client.patch(f"{API}/menu-items/{menu['dosa']}", json={"price_paise": 99999})
        retry = _place(live_client, menu["restaurant"], body)
        assert retry.status_code == 200
        assert retry.json()["total_paise"] == first["total_paise"] == 12050


class TestSnapshots:
    def test_menu_edits_do_not_change_existing_orders(self, live_client, menu) -> None:
        order = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 2)])).json()
        live_client.patch(
            f"{API}/menu-items/{menu['dosa']}", json={"name": "Ghee Roast Dosa", "price_paise": 20000}
        )
        fetched = live_client.get(f"{API}/orders/{order['id']}").json()
        (line,) = fetched["items"]
        assert (line["item_name"], line["unit_price_paise"], line["line_total_paise"]) == (
            "Masala Dosa", 12050, 24100,
        )
        assert fetched["total_paise"] == 24100

        # New orders use the new price.
        new = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 2)])).json()
        assert new["items"][0]["item_name"] == "Ghee Roast Dosa"
        assert new["total_paise"] == 40000

    def test_deleting_menu_item_keeps_order_history(self, live_client, menu) -> None:
        order = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)])).json()
        assert live_client.delete(f"{API}/menu-items/{menu['dosa']}").status_code == 204
        (line,) = live_client.get(f"{API}/orders/{order['id']}").json()["items"]
        assert line["menu_item_id"] is None
        assert (line["item_name"], line["unit_price_paise"]) == ("Masala Dosa", 12050)


class TestStatusTransitions:
    def _status(self, client, order_id, status):
        if status == "confirmed":  # orders are confirmed by their (demo) payment
            return _pay(client, order_id)
        return client.patch(f"{API}/orders/{order_id}/status", json={"status": status})

    def test_pending_cannot_be_confirmed_without_payment(self, live_client, menu) -> None:
        order = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)])).json()
        response = live_client.patch(f"{API}/orders/{order['id']}/status", json={"status": "confirmed"})
        assert response.status_code == 409
        assert "payment" in response.json()["detail"]
        assert live_client.get(f"{API}/orders/{order['id']}").json()["status"] == "pending"

    def test_full_happy_path(self, live_client, menu) -> None:
        order = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)])).json()
        for status in ("confirmed", "preparing", "ready", "completed"):
            response = self._status(live_client, order["id"], status)
            assert response.status_code == 200, response.text
            assert live_client.get(f"{API}/orders/{order['id']}").json()["status"] == status
        final = live_client.get(f"{API}/orders/{order['id']}").json()
        assert final["status"] == "completed" and final["payment"]["status"] == "succeeded"

    @pytest.mark.parametrize(
        ("path", "invalid"),
        [
            ([], "preparing"),                                   # pending -> preparing skips a step
            ([], "completed"),
            (["confirmed", "preparing", "ready", "completed"], "pending"),
            (["confirmed", "preparing", "ready", "completed"], "cancelled"),
            (["cancelled"], "preparing"),
            (["cancelled"], "pending"),
        ],
    )
    def test_invalid_transitions_return_409(self, live_client, menu, path, invalid) -> None:
        order = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)])).json()
        for status in path:
            assert self._status(live_client, order["id"], status).status_code == 200
        before = live_client.get(f"{API}/orders/{order['id']}").json()["status"]
        response = self._status(live_client, order["id"], invalid)
        assert response.status_code == 409
        assert "Cannot change order status" in response.json()["detail"]
        assert live_client.get(f"{API}/orders/{order['id']}").json()["status"] == before

    def test_cancel_from_preparing(self, live_client, menu) -> None:
        order = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)])).json()
        for status in ("confirmed", "preparing", "cancelled"):
            assert self._status(live_client, order["id"], status).status_code == 200

    def test_same_status_is_a_no_op(self, live_client, menu) -> None:
        order = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)])).json()
        response = self._status(live_client, order["id"], "pending")
        assert response.status_code == 200
        assert response.json()["updated_at"] == order["updated_at"]

    def test_missing_order_returns_404(self, live_client) -> None:
        assert live_client.get(f"{API}/orders/{uuid.uuid4()}").status_code == 404
        assert self._status(live_client, uuid.uuid4(), "confirmed").status_code == 404


class TestListOrders:
    def test_pagination_newest_first_and_scoped(self, live_client, menu) -> None:
        ids = [
            _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], n)])).json()["id"]
            for n in range(1, 6)
        ]
        other = _restaurant(live_client)
        other_item = _item(live_client, other, "Idli", 3000)
        _place(live_client, other, _order_body([(other_item, 1)]))

        url = f"{API}/restaurants/{menu['restaurant']}/orders"
        page1 = live_client.get(url, params={"limit": 2}).json()
        page2 = live_client.get(url, params={"limit": 2, "offset": 2}).json()
        page3 = live_client.get(url, params={"limit": 2, "offset": 4}).json()
        assert page1["total"] == 5
        listed = [o["id"] for p in (page1, page2, page3) for o in p["items"]]
        assert listed == list(reversed(ids))
        assert all(o["items"] for o in page1["items"])  # items included

    def test_status_filter(self, live_client, menu) -> None:
        url = f"{API}/restaurants/{menu['restaurant']}/orders"
        a = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)])).json()
        b = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 2)])).json()
        assert _pay(live_client, b["id"]).status_code == 200

        pending = live_client.get(url, params={"status": "pending"}).json()
        confirmed = live_client.get(url, params={"status": "confirmed"}).json()
        cancelled = live_client.get(url, params={"status": "cancelled"}).json()
        assert [o["id"] for o in pending["items"]] == [a["id"]]
        assert [o["id"] for o in confirmed["items"]] == [b["id"]]
        assert cancelled == {"items": [], "total": 0, "limit": 20, "offset": 0}

    def test_unknown_restaurant_returns_404(self, live_client) -> None:
        assert live_client.get(f"{API}/restaurants/{uuid.uuid4()}/orders").status_code == 404


class TestTransactionRollback:
    def test_failure_while_inserting_lines_rolls_back_whole_order(
        self, live_client, menu, connection, monkeypatch
    ) -> None:
        """Force the second line to violate a database constraint: the order
        row (inserted first) must be rolled back together with the lines."""
        from app.services import orders as order_service

        real_price_lines = order_service.price_lines

        def corrupt_second_line(menu_items, lines):
            priced = real_price_lines(menu_items, lines)
            bad = priced[1]
            priced[1] = order_service.PricedLine(bad.menu_item_id, bad.item_name, bad.unit_price_paise, 0)
            return priced

        monkeypatch.setattr(order_service, "price_lines", corrupt_second_line)
        orders_before = _order_count(connection)
        lines_before = connection.execute(select(func.count()).select_from(OrderItem)).scalar_one()

        response = _place(
            live_client, menu["restaurant"],
            _order_body([(menu["dosa"], 1), (menu["coffee"], 1)], idempotency_key="partial"),
        )  # fmt: skip
        assert response.status_code == 409
        assert response.json() == {"detail": "quantity must be a positive integer."}
        assert _order_count(connection) == orders_before
        assert connection.execute(select(func.count()).select_from(OrderItem)).scalar_one() == lines_before

        # The key was not consumed: a correct retry succeeds.
        monkeypatch.setattr(order_service, "price_lines", real_price_lines)
        retry = _place(
            live_client, menu["restaurant"],
            _order_body([(menu["dosa"], 1), (menu["coffee"], 1)], idempotency_key="partial"),
        )  # fmt: skip
        assert retry.status_code == 201


class TestPrivacyAndRegression:
    def test_logs_and_errors_exclude_customer_details(self, live_client, menu, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger="app"):
            created = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 1)], idempotency_key="pii"))
            conflict = _place(live_client, menu["restaurant"], _order_body([(menu["dosa"], 2)], idempotency_key="pii"))
            _pay(live_client, created.json()["id"])
        assert f"Created order {created.json()['order_number']}" in caplog.text  # logging is live
        assert PHONE not in caplog.text and "Asha" not in caplog.text
        assert PHONE not in conflict.text and "Asha" not in conflict.text

    def test_existing_apis_still_work(self, live_client, menu) -> None:
        assert live_client.get("/health").status_code == 200
        assert live_client.get("/health/db").json() == {"status": "ok", "database": "connected"}
        assert live_client.get(f"{API}/restaurants/{menu['restaurant']}").status_code == 200
        items = live_client.get(f"{API}/restaurants/{menu['restaurant']}/menu-items").json()
        assert items["total"] == 3
