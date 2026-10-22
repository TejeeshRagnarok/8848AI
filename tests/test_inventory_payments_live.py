"""Live tests for stock, reservations, demo payments, expiry and cancellation,
plus the agent's database-backed tools.  Real PostgreSQL (``live_db``); every
test runs in the rolled-back session from ``conftest.py``.  The LLM is the
scripted fake.  Concurrency with committed data is in
``test_orders_concurrency_live.py``.
"""

import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import pytest
from db_helpers import count_test_orders, count_test_payments
from agent_fakes import ScriptedProvider, call, last_tool_result, say
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Connection, func, select, update

from app.agent.sessions import InMemorySessionStore
from app.config import settings
from app.main import app
from app.models import InventoryItem, InventoryMovement, Order, Payment
from app.routers.agent import get_provider_factory, get_session_store
from app.security import require_staff

pytestmark = pytest.mark.live_db

API = "/api/v1"
PHONE = "+91 98765 43210"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _restaurant(client: TestClient) -> str:
    r = client.post(f"{API}/restaurants", json={"name": "Stock Kitchen", "slug": f"test-{uuid.uuid4().hex[:12]}"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _item(client: TestClient, rid: str, name: str, price: int, stock: int | None = None, **extra) -> str:
    r = client.post(f"{API}/restaurants/{rid}/menu-items", json={"name": name, "price_paise": price} | extra)
    assert r.status_code == 201, r.text
    item_id = r.json()["id"]
    if stock is not None:
        s = client.put(f"{API}/menu-items/{item_id}/inventory", json={"on_hand": stock})
        assert s.status_code == 200, s.text
    return item_id


@pytest.fixture
def menu(live_client: TestClient) -> dict:
    rid = _restaurant(live_client)
    return {
        "restaurant": rid,
        "dosa": _item(live_client, rid, "Masala Dosa", 8000, stock=5, category="Dosa"),
        "plain": _item(live_client, rid, "Plain Dosa", 6000, stock=10, category="Dosa"),
        "coffee": _item(live_client, rid, "Filter Coffee", 2500, stock=20, category="Beverages"),
        "vada": _item(live_client, rid, "Medu Vada", 5000, stock=0, category="Snacks"),
        "water": _item(live_client, rid, "Water Bottle", 2000, category="Beverages"),  # untracked
    }


def _order(client: TestClient, rid: str, lines: list[tuple[str, int]], **extra):
    body = {"customer_phone": PHONE, "items": [{"menu_item_id": i, "quantity": q} for i, q in lines]}
    return client.post(f"{API}/restaurants/{rid}/orders", json=body | extra)


def _stock(connection: Connection, item_id: str) -> tuple[int, int, int]:
    row = connection.execute(
        select(InventoryItem.on_hand, InventoryItem.reserved, InventoryItem.sold)
        .where(InventoryItem.menu_item_id == uuid.UUID(item_id))
    ).one()
    return tuple(row)


def _movements(connection: Connection, order_id: str) -> list[tuple[str, int]]:
    rows = connection.execute(
        select(InventoryMovement.kind, InventoryMovement.quantity)
        .where(InventoryMovement.order_id == uuid.UUID(order_id))
        .order_by(InventoryMovement.created_at, InventoryMovement.kind)
    ).all()
    return sorted(tuple(r) for r in rows)


def _simulate(client: TestClient, order: dict, outcome: str = "succeeded", amount: int | None = None):
    payment = order["payment"]
    return client.post(
        f"{API}/payments/{payment['id']}/simulate",
        json={"outcome": outcome, "amount_paise": payment["amount_paise"] if amount is None else amount},
    )


def _get(client: TestClient, order_id: str) -> dict:
    return client.get(f"{API}/orders/{order_id}").json()


def _expire_now(connection: Connection, order_id: str) -> None:
    connection.execute(
        update(Order).where(Order.id == uuid.UUID(order_id))
        .values(reservation_expires_at=datetime.now(UTC) - timedelta(seconds=1))
    )  # fmt: skip


def _token_headers(order: dict) -> dict:
    return {"X-Order-Token": order["customer_token"]}


# ── Menu and stock lookup ────────────────────────────────────────────────────


class TestMenuStock:
    def test_menu_shows_live_stock(self, live_client, menu) -> None:
        items = {i["name"]: i for i in live_client.get(
            f"{API}/restaurants/{menu['restaurant']}/menu-items").json()["items"]}  # fmt: skip
        assert (items["Masala Dosa"]["available_quantity"], items["Masala Dosa"]["orderable"]) == (5, True)
        assert (items["Medu Vada"]["available_quantity"], items["Medu Vada"]["orderable"]) == (0, False)
        assert items["Medu Vada"]["is_available"] is True  # switched on, but sold out
        assert (items["Water Bottle"]["stock_tracked"], items["Water Bottle"]["available_quantity"]) == (False, None)
        in_stock = live_client.get(f"{API}/restaurants/{menu['restaurant']}/menu-items",
                                   params={"in_stock": True}).json()  # fmt: skip
        assert "Medu Vada" not in {i["name"] for i in in_stock["items"]}
        assert in_stock["total"] == 4

    def test_inventory_view_and_ledger(self, live_client, menu) -> None:
        stock = {s["name"]: s for s in live_client.get(f"{API}/restaurants/{menu['restaurant']}/inventory").json()}
        assert stock["Masala Dosa"] | {} == stock["Masala Dosa"]
        assert (stock["Masala Dosa"]["on_hand"], stock["Masala Dosa"]["reserved"],
                stock["Masala Dosa"]["available"], stock["Masala Dosa"]["sold"]) == (5, 0, 5, 0)  # fmt: skip
        assert stock["Water Bottle"]["stock_tracked"] is False
        ledger = live_client.get(f"{API}/menu-items/{menu['dosa']}/inventory/movements").json()
        assert [(m["kind"], m["quantity"], m["on_hand_after"]) for m in ledger] == [("restock", 5, 5)]

    def test_restock_and_stock_take(self, live_client, menu, connection) -> None:
        assert live_client.put(f"{API}/menu-items/{menu['vada']}/inventory", json={"on_hand": 12}).status_code == 200
        assert _stock(connection, menu["vada"]) == (12, 0, 0)
        live_client.put(f"{API}/menu-items/{menu['vada']}/inventory", json={"on_hand": 9, "note": "count"})
        kinds = [m["kind"] for m in live_client.get(f"{API}/menu-items/{menu['vada']}/inventory/movements").json()]
        assert kinds[:2] == ["adjust", "restock"]

    @pytest.mark.parametrize("body", [{"on_hand": -1}, {"on_hand": 1.5}, {"on_hand": "3"}, {}])
    def test_invalid_stock_updates_rejected(self, live_client, menu, body) -> None:
        assert live_client.put(f"{API}/menu-items/{menu['dosa']}/inventory", json=body).status_code == 422

    def test_stock_cannot_drop_below_reserved(self, live_client, menu) -> None:
        assert _order(live_client, menu["restaurant"], [(menu["dosa"], 3)]).status_code == 201
        r = live_client.put(f"{API}/menu-items/{menu['dosa']}/inventory", json={"on_hand": 2})
        assert r.status_code == 409 and "reserved" in r.json()["detail"]

    def test_item_with_reservations_cannot_be_deleted(self, live_client, menu) -> None:
        _order(live_client, menu["restaurant"], [(menu["dosa"], 1)])
        assert live_client.delete(f"{API}/menu-items/{menu['dosa']}").status_code == 409


# ── Placing orders: validation, pricing, reservation ─────────────────────────


class TestPlaceAndReserve:
    def test_order_reserves_stock_with_db_prices(self, live_client, menu, connection) -> None:
        r = _order(live_client, menu["restaurant"], [(menu["dosa"], 2), (menu["coffee"], 1), (menu["water"], 3)])
        assert r.status_code == 201, r.text
        order = r.json()
        assert order["total_paise"] == 2 * 8000 + 2500 + 3 * 2000
        assert (order["status"], order["inventory_state"]) == ("pending", "reserved")
        expires = datetime.fromisoformat(order["reservation_expires_at"])
        assert timedelta(0) < expires - datetime.now(UTC) <= timedelta(seconds=settings.order_reservation_ttl_seconds)
        assert order["payment"]["status"] == "pending" and order["payment"]["amount_paise"] == order["total_paise"]
        assert _stock(connection, menu["dosa"]) == (5, 2, 0)
        assert _stock(connection, menu["coffee"]) == (20, 1, 0)
        assert _movements(connection, order["id"]) == [("reserve", 1), ("reserve", 2)]

    def test_insufficient_stock_rolls_back_everything(self, live_client, menu, connection) -> None:
        r = _order(live_client, menu["restaurant"], [(menu["coffee"], 1), (menu["dosa"], 6)])
        assert r.status_code == 409
        assert "only 5 of 'Masala Dosa' available (requested 6)" in r.json()["detail"]
        assert count_test_orders(connection) == 0
        assert count_test_payments(connection) == 0
        assert _stock(connection, menu["coffee"]) == (20, 0, 0)  # first line's reservation undone

    def test_sold_out_item_rejected(self, live_client, menu) -> None:
        assert _order(live_client, menu["restaurant"], [(menu["vada"], 1)]).status_code == 409

    @pytest.mark.parametrize("quantity", [0, -2, 101, 1.5, "2"])
    def test_invalid_quantities_rejected(self, live_client, menu, connection, quantity) -> None:
        assert _order(live_client, menu["restaurant"], [(menu["dosa"], quantity)]).status_code == 422
        assert _stock(connection, menu["dosa"]) == (5, 0, 0)

    def test_unknown_item_rejected(self, live_client, menu) -> None:
        r = _order(live_client, menu["restaurant"], [(str(uuid.uuid4()), 1)])
        assert r.status_code == 422 and r.json()["detail"][0]["type"] == "menu_item_not_found"

    def test_client_cannot_supply_prices_or_payment(self, live_client, menu) -> None:
        for extra in ({"total_paise": 1}, {"payment_status": "succeeded"}, {"status": "confirmed"}):
            assert _order(live_client, menu["restaurant"], [(menu["dosa"], 1)], **extra).status_code == 422

    def test_idempotent_retry_reserves_once(self, live_client, menu, connection) -> None:
        a = _order(live_client, menu["restaurant"], [(menu["dosa"], 2)], idempotency_key="k-1")
        b = _order(live_client, menu["restaurant"], [(menu["dosa"], 2)], idempotency_key="k-1")
        assert (a.status_code, b.status_code) == (201, 200)
        assert a.json()["id"] == b.json()["id"]
        assert _stock(connection, menu["dosa"]) == (5, 2, 0)
        assert count_test_payments(connection) == 1


# ── Demo payments ────────────────────────────────────────────────────────────


class TestDemoPayment:
    def test_success_confirms_and_sells_stock_once(self, live_client, menu, connection) -> None:
        order = _order(live_client, menu["restaurant"], [(menu["dosa"], 2)]).json()
        r = _simulate(live_client, order)
        assert r.status_code == 200, r.text
        payment = r.json()
        assert (payment["status"], payment["mode"], payment["simulated"]) == ("succeeded", "demo", True)
        assert "no real money" in payment["notice"]
        fetched = _get(live_client, order["id"])
        assert (fetched["status"], fetched["inventory_state"], fetched["reservation_expires_at"]) == (
            "confirmed", "sold", None)  # fmt: skip
        assert _stock(connection, menu["dosa"]) == (3, 0, 2)

        again = _simulate(live_client, order)
        assert again.status_code == 200 and again.headers["Idempotent-Replayed"] == "true"
        assert _stock(connection, menu["dosa"]) == (3, 0, 2)
        assert _movements(connection, order["id"]) == [("reserve", 2), ("sell", 2)]

    def test_failure_cancels_and_releases(self, live_client, menu, connection) -> None:
        order = _order(live_client, menu["restaurant"], [(menu["dosa"], 2)]).json()
        r = live_client.post(f"{API}/payments/{order['payment']['id']}/simulate",
                             json={"outcome": "failed", "amount_paise": 16000, "reason": "Card declined (demo)"})  # fmt: skip
        assert r.status_code == 200 and r.json()["status"] == "failed"
        assert r.json()["failure_reason"] == "Card declined (demo)"
        fetched = _get(live_client, order["id"])
        assert (fetched["status"], fetched["cancel_reason"], fetched["inventory_state"]) == (
            "cancelled", "payment_failed", "released")  # fmt: skip
        assert _stock(connection, menu["dosa"]) == (5, 0, 0)
        # A failed payment can never become a success, and failing again changes nothing.
        assert _simulate(live_client, order).status_code == 409
        assert _simulate(live_client, order, "failed").headers["Idempotent-Replayed"] == "true"
        assert _movements(connection, order["id"]) == [("release", 2), ("reserve", 2)]

    @pytest.mark.parametrize("amount", [1, 15999, 16001])
    def test_amount_mismatch_changes_nothing(self, live_client, menu, connection, amount) -> None:
        order = _order(live_client, menu["restaurant"], [(menu["dosa"], 2)]).json()
        r = _simulate(live_client, order, amount=amount)
        assert r.status_code == 422 and r.json()["detail"][0]["type"] == "amount_mismatch"
        assert _get(live_client, order["id"])["payment"]["status"] == "pending"
        assert _stock(connection, menu["dosa"]) == (5, 2, 0)

    def test_expired_reservation_cannot_be_paid(self, live_client, menu, connection) -> None:
        order = _order(live_client, menu["restaurant"], [(menu["dosa"], 2)]).json()
        _expire_now(connection, order["id"])
        r = _simulate(live_client, order)
        assert r.status_code == 409 and "expired" in r.json()["detail"]
        fetched = _get(live_client, order["id"])
        assert (fetched["status"], fetched["cancel_reason"], fetched["payment"]["status"]) == (
            "cancelled", "expired", "expired")  # fmt: skip
        assert _stock(connection, menu["dosa"]) == (5, 0, 0)

    def test_unknown_payment_404_and_invalid_outcome_422(self, live_client, menu) -> None:
        r = live_client.post(f"{API}/payments/{uuid.uuid4()}/simulate", json={"outcome": "succeeded", "amount_paise": 1})
        assert r.status_code == 404
        order = _order(live_client, menu["restaurant"], [(menu["dosa"], 1)]).json()
        for outcome in ("paid", "refunded", "pending"):
            assert _simulate(live_client, order, outcome).status_code == 422


# ── Reservation expiry ───────────────────────────────────────────────────────


class TestExpiry:
    def test_expiry_releases_stock_exactly_once(self, live_client, menu, connection) -> None:
        old = _order(live_client, menu["restaurant"], [(menu["dosa"], 3)]).json()
        fresh = _order(live_client, menu["restaurant"], [(menu["dosa"], 1)]).json()
        _expire_now(connection, old["id"])

        first = live_client.post(f"{API}/maintenance/expire-reservations").json()
        second = live_client.post(f"{API}/maintenance/expire-reservations").json()
        assert first == {"expired_orders": [old["order_number"]]} and second == {"expired_orders": []}
        assert _stock(connection, menu["dosa"]) == (5, 1, 0)  # only the fresh order still holds stock
        assert _movements(connection, old["id"]) == [("release", 3), ("reserve", 3)]
        assert _get(live_client, fresh["id"])["status"] == "pending"
        expired = _get(live_client, old["id"])
        assert (expired["status"], expired["payment"]["status"]) == ("cancelled", "expired")


# ── Cancellation, tracking and history ───────────────────────────────────────


class TestCancellation:
    def test_customer_cancel_is_idempotent(self, live_client, menu, connection) -> None:
        order = _order(live_client, menu["restaurant"], [(menu["dosa"], 2)]).json()
        url = f"{API}/orders/{order['id']}/cancel"
        a = live_client.post(url, headers=_token_headers(order))
        b = live_client.post(url, headers=_token_headers(order))
        assert (a.status_code, b.status_code) == (200, 200)
        assert a.json()["status"] == "cancelled" and a.json()["cancel_reason"] == "customer"
        assert a.json()["payment"]["status"] == "cancelled"
        assert _stock(connection, menu["dosa"]) == (5, 0, 0)
        assert _movements(connection, order["id"]) == [("release", 2), ("reserve", 2)]

    def test_cancel_paid_order_returns_stock_and_refunds(self, live_client, menu, connection) -> None:
        order = _order(live_client, menu["restaurant"], [(menu["dosa"], 2)]).json()
        _simulate(live_client, order)
        r = live_client.post(f"{API}/orders/{order['id']}/cancel", headers=_token_headers(order))
        assert r.status_code == 200
        assert (r.json()["payment"]["status"], _stock(connection, menu["dosa"])) == ("refunded", (5, 0, 0))
        assert _get(live_client, order["id"])["inventory_state"] == "returned"

    def test_customer_cannot_cancel_after_preparation_starts(self, live_client, menu, connection) -> None:
        order = _order(live_client, menu["restaurant"], [(menu["dosa"], 1)]).json()
        _simulate(live_client, order)
        live_client.patch(f"{API}/orders/{order['id']}/status", json={"status": "preparing"})
        r = live_client.post(f"{API}/orders/{order['id']}/cancel", headers=_token_headers(order))
        assert r.status_code == 409 and "preparation" in r.json()["detail"]
        # Staff can; the food is considered used, so stock stays sold, and the
        # payment is refunded (demo).
        s = live_client.patch(f"{API}/orders/{order['id']}/status", json={"status": "cancelled"})
        assert s.status_code == 200 and s.json()["payment"]["status"] == "refunded"
        assert _stock(connection, menu["dosa"]) == (4, 0, 1)

    def test_wrong_token_cannot_see_or_cancel(self, live_client, menu) -> None:
        a = _order(live_client, menu["restaurant"], [(menu["dosa"], 1)]).json()
        b = _order(live_client, menu["restaurant"], [(menu["coffee"], 1)]).json()
        for path in ("track", "cancel"):
            method = live_client.get if path == "track" else live_client.post
            assert method(f"{API}/orders/{a['id']}/{path}", headers=_token_headers(b)).status_code == 404
            assert method(f"{API}/orders/{a['id']}/{path}").status_code == 404
        assert _get(live_client, a["id"])["status"] == "pending"

    def test_tracking_hides_contact_details(self, live_client, menu) -> None:
        order = _order(live_client, menu["restaurant"], [(menu["dosa"], 1)], customer_name="Asha").json()
        tracked = live_client.get(f"{API}/orders/{order['id']}/track", headers=_token_headers(order))
        assert tracked.status_code == 200
        assert tracked.json()["payment"]["status"] == "pending"
        assert PHONE not in tracked.text and "Asha" not in tracked.text


class TestHistory:
    def test_completed_order_keeps_history_and_sells_once(self, live_client, menu, connection) -> None:
        order = _order(live_client, menu["restaurant"], [(menu["dosa"], 2), (menu["coffee"], 1)]).json()
        _simulate(live_client, order)
        for status in ("preparing", "ready", "completed"):
            assert live_client.patch(f"{API}/orders/{order['id']}/status", json={"status": status}).status_code == 200
        final = _get(live_client, order["id"])
        assert final["status"] == "completed" and len(final["items"]) == 2
        assert final["payment"]["status"] == "succeeded"
        assert _stock(connection, menu["dosa"]) == (3, 0, 2)
        assert _movements(connection, order["id"]) == [("reserve", 1), ("reserve", 2), ("sell", 1), ("sell", 2)]
        assert live_client.patch(f"{API}/orders/{order['id']}/status", json={"status": "cancelled"}).status_code == 409


# ── Authentication ───────────────────────────────────────────────────────────


class TestStaffAuth:
    @pytest.fixture
    def real_auth(self, monkeypatch):
        app.dependency_overrides.pop(require_staff, None)
        monkeypatch.setattr(settings, "staff_api_key", SecretStr("demo-staff-key"))

    def test_payment_status_cannot_be_changed_without_staff_key(self, live_client, menu, real_auth, connection) -> None:
        order = _order(live_client, menu["restaurant"], [(menu["dosa"], 1)]).json()  # public
        url = f"{API}/payments/{order['payment']['id']}/simulate"
        body = {"outcome": "succeeded", "amount_paise": 8000}
        assert live_client.post(url, json=body).status_code == 401
        assert live_client.post(url, json=body, headers={"X-Staff-Key": "wrong"}).status_code == 401
        assert live_client.post(url, json=body, headers=_token_headers(order)).status_code == 401
        assert connection.execute(select(Payment.status).where(Payment.id == uuid.UUID(order["payment"]["id"]))).scalar_one() == "pending"
        ok = live_client.post(url, json=body, headers={"X-Staff-Key": "demo-staff-key"})
        assert ok.status_code == 200 and ok.json()["status"] == "succeeded"

    @pytest.mark.parametrize(("method", "path"), [
        ("get", "/orders/{order}"), ("patch", "/orders/{order}/status"),
        ("get", "/restaurants/{rid}/orders"), ("get", "/restaurants/{rid}/inventory"),
        ("put", "/menu-items/{item}/inventory"), ("post", "/maintenance/expire-reservations"),
        ("post", "/restaurants/{rid}/menu-items"), ("patch", "/menu-items/{item}"),
        ("get", "/orders/{order}/payment"),
    ])  # fmt: skip
    def test_staff_endpoints_reject_anonymous(self, live_client, menu, real_auth, method, path) -> None:
        order = _order(live_client, menu["restaurant"], [(menu["dosa"], 1)]).json()
        url = API + path.format(order=order["id"], rid=menu["restaurant"], item=menu["dosa"])
        kwargs = {} if method == "get" else {"json": {}}
        assert live_client.request(method.upper(), url, **kwargs).status_code == 401

    def test_staff_endpoints_disabled_without_configured_key(self, live_client, menu, monkeypatch) -> None:
        app.dependency_overrides.pop(require_staff, None)
        monkeypatch.setattr(settings, "staff_api_key", None)
        r = live_client.post(f"{API}/maintenance/expire-reservations", headers={"X-Staff-Key": "anything"})
        assert r.status_code == 503

    def test_public_endpoints_stay_public(self, live_client, menu, real_auth) -> None:
        assert live_client.get(f"{API}/restaurants/{menu['restaurant']}/menu-items").status_code == 200
        assert _order(live_client, menu["restaurant"], [(menu["dosa"], 1)]).status_code == 201


# ── Agent tools on real data ─────────────────────────────────────────────────


@pytest.fixture
def agent(live_client: TestClient) -> Generator:
    holder: dict = {}
    store = InMemorySessionStore(ttl_seconds=600, max_sessions=50)
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_provider_factory] = lambda: (lambda: holder["provider"])

    def install(provider: ScriptedProvider) -> TestClient:
        holder["provider"] = provider
        return live_client

    try:
        yield install
    finally:
        app.dependency_overrides.pop(get_session_store, None)
        app.dependency_overrides.pop(get_provider_factory, None)


def _chat(client: TestClient, rid: str, **body):
    return client.post(f"{API}/restaurants/{rid}/agent/chat", json=body)


class TestAgentWithStock:
    def test_get_menu_hides_sold_out_items_and_shows_stock(self, agent, menu) -> None:
        provider = ScriptedProvider(call("get_menu"), say("ok"))
        _chat(agent(provider), menu["restaurant"], message="What do you have?")
        items = {i["name"]: i for i in last_tool_result(provider.requests[1]["messages"])["items"]}
        assert "Medu Vada" not in items
        assert items["Masala Dosa"]["available_quantity"] == 5 and items["Water Bottle"]["available_quantity"] is None

    def test_short_stock_is_explained_with_real_alternatives(self, agent, menu, connection) -> None:
        provider = ScriptedProvider(
            call("create_order", customer_phone=PHONE,
                 items=[{"menu_item_id": menu["dosa"], "quantity": 9}, {"menu_item_id": menu["vada"], "quantity": 1}]),
            say("Sorry, we only have 5 masala dosas, and medu vada is sold out."),
        )  # fmt: skip
        body = _chat(agent(provider), menu["restaurant"], message="9 masala dosa and a vada").json()
        result = last_tool_result(provider.requests[1]["messages"])
        assert result["error"] == "items_not_orderable"
        problems = {p["menu_item_id"]: p for p in result["problems"]}
        assert problems[menu["dosa"]]["problem"] == "insufficient_stock"
        assert problems[menu["dosa"]]["available_quantity"] == 5
        assert problems[menu["vada"]]["problem"] == "insufficient_stock"
        alternatives = {a["name"] for a in result["alternatives"]}
        assert alternatives == {"Plain Dosa", "Filter Coffee", "Water Bottle"}  # in stock, not requested
        assert body["requires_confirmation"] is False
        assert _stock(connection, menu["dosa"]) == (5, 0, 0)

    def test_cart_changes_reserve_nothing_until_confirmed(self, agent, menu, connection) -> None:
        provider = ScriptedProvider(
            call("create_order", customer_phone=PHONE, items=[
                {"menu_item_id": menu["dosa"], "quantity": 2}, {"menu_item_id": menu["coffee"], "quantity": 1}]),
            say("Two masala dosas and a coffee, INR 185.00. Anything else?"),
            call("create_order", customer_phone=PHONE, items=[
                {"menu_item_id": menu["dosa"], "quantity": 2}, {"menu_item_id": menu["coffee"], "quantity": 1},
                {"menu_item_id": menu["plain"], "quantity": 1}]),
            say("Added a plain dosa. The total is INR 245.00. Shall I place it?"),
        )  # fmt: skip
        client = agent(provider)
        first = _chat(client, menu["restaurant"], message="Two masala dosas and one coffee").json()
        second = _chat(client, menu["restaurant"], session_id=first["session_id"], message="Add one plain dosa").json()
        assert first["pending_order"]["total_paise"] == 18500
        assert second["pending_order"]["total_paise"] == 24500
        assert second["pending_order"]["proposal_id"] != first["pending_order"]["proposal_id"]
        assert count_test_orders(connection) == 0
        assert _stock(connection, menu["dosa"]) == (5, 0, 0)  # nothing reserved by cart edits

        placed = _chat(client, menu["restaurant"], session_id=first["session_id"],
                       confirm_proposal_id=second["pending_order"]["proposal_id"]).json()  # fmt: skip
        assert placed["order"]["total_paise"] == 24500 and placed["order"]["inventory_state"] == "reserved"
        assert "demo payment is pending" in placed["reply"]
        assert _stock(connection, menu["dosa"]) == (5, 2, 0) and _stock(connection, menu["plain"]) == (10, 1, 0)

    def test_confirmation_fails_cleanly_when_stock_ran_out(self, agent, menu, connection, live_client) -> None:
        provider = ScriptedProvider(
            call("create_order", customer_phone=PHONE, items=[{"menu_item_id": menu["dosa"], "quantity": 4}]),
            say("Four masala dosas, INR 320.00. Shall I place it?"),
        )  # fmt: skip
        client = agent(provider)
        first = _chat(client, menu["restaurant"], message="Four masala dosas").json()
        _order(live_client, menu["restaurant"], [(menu["dosa"], 3)])  # someone else takes 3 of 5
        placed = _chat(client, menu["restaurant"], session_id=first["session_id"],
                       confirm_proposal_id=first["pending_order"]["proposal_id"]).json()  # fmt: skip
        assert placed["order"] is None
        assert "only 2 of 'Masala Dosa' available" in placed["reply"]
        assert _stock(connection, menu["dosa"]) == (5, 3, 0)

    def test_payment_claims_follow_the_database(self, agent, menu, live_client) -> None:
        provider = ScriptedProvider(
            call("create_order", customer_phone=PHONE, items=[{"menu_item_id": menu["dosa"], "quantity": 1}]),
            say("One masala dosa, INR 80.00. Shall I place it?"),
            # Customer: "I paid already" -> the model checks, but claims success anyway.
            call("get_order_status"),
            say("Great, your payment has been received!"),
            # After the (staff-simulated) payment really succeeded:
            call("get_order_status"),
            say("Your payment has been received and your order is confirmed."),
        )  # fmt: skip
        client = agent(provider)
        first = _chat(client, menu["restaurant"], message="One masala dosa").json()
        placed = _chat(client, menu["restaurant"], session_id=first["session_id"],
                       confirm_proposal_id=first["pending_order"]["proposal_id"]).json()  # fmt: skip

        claim = _chat(client, menu["restaurant"], session_id=first["session_id"], message="I paid already").json()
        # requests: [0] create_order, [1] its reply, [2] get_order_status, [3] reply after it.
        status = last_tool_result(provider.requests[3]["messages"])
        assert status["orders"][0]["payment_status"] == "pending"
        assert "payment has been received" not in claim["reply"]
        assert _get(live_client, placed["order"]["id"])["payment"]["status"] == "pending"

        order = _get(live_client, placed["order"]["id"])
        _simulate(live_client, order)
        after = _chat(client, menu["restaurant"], session_id=first["session_id"], message="Status?").json()
        assert last_tool_result(provider.requests[5]["messages"])["orders"][0]["payment_status"] == "succeeded"
        assert after["reply"] == "Your payment has been received and your order is confirmed."

    def test_order_status_is_limited_to_this_conversation(self, agent, menu, live_client) -> None:
        other = _order(live_client, menu["restaurant"], [(menu["dosa"], 1)]).json()
        provider = ScriptedProvider(call("get_order_status", order_number=other["order_number"]), say("Not found."))
        _chat(agent(provider), menu["restaurant"], message=f"Status of {other['order_number']}?")
        assert last_tool_result(provider.requests[1]["messages"])["error"] == "no_orders"
