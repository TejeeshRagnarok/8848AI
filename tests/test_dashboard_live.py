"""Restaurant dashboard against real PostgreSQL (``live_db``).

Covers sign-in and restaurant scoping, menu and stock management, order
channels, walk-in sales, reservation expiry, history after retiring a dish,
and dashboard changes being what the voice/chat agent sees next.  Every test
runs in the rolled-back session from ``conftest.py``.
"""

import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import pytest
from agent_fakes import ScriptedProvider, call, last_tool_result, say
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select, update

from app.agent.sessions import InMemorySessionStore
from app.config import settings
from app.main import app
from app.models import InventoryMovement, Order
from app.routers.agent import get_provider_factory, get_session_store
from app.security import DASHBOARD_COOKIE, login_throttle
from app.services import dashboard as dashboard_service

pytestmark = pytest.mark.live_db

API = "/api/v1"
D = "/api/v1/dashboard"
KEY = "dashboard-test-staff-key"
HDR = {"X-Requested-With": "dashboard"}
PHONE = "+91 90000 00001"


@pytest.fixture(autouse=True)
def staff_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "staff_api_key", SecretStr(KEY))
    login_throttle._failures.clear()


def _restaurant(client: TestClient, name: str = "Dash Kitchen") -> dict:
    r = client.post(f"{API}/restaurants", json={"name": name, "slug": f"test-{uuid.uuid4().hex[:12]}"})
    assert r.status_code == 201, r.text
    return r.json()


def _item(client: TestClient, rid: str, name: str, price: int, stock: int | None = None, **extra) -> str:
    r = client.post(f"{API}/restaurants/{rid}/menu-items", json={"name": name, "price_paise": price} | extra)
    assert r.status_code == 201, r.text
    if stock is not None:
        assert client.put(f"{API}/menu-items/{r.json()['id']}/inventory", json={"on_hand": stock}).status_code == 200
    return r.json()["id"]


def _sign_in(client: TestClient, slug: str) -> dict:
    r = client.post(f"{D}/session", json={"staff_key": KEY, "restaurant_slug": slug})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
def shop(live_client: TestClient) -> dict:
    restaurant = _restaurant(live_client)
    rid = restaurant["id"]
    data = {
        "restaurant": rid, "slug": restaurant["slug"],
        "dosa": _item(live_client, rid, "Masala Dosa", 8000, 5, category="Dosa", description="Potato filling."),
        "coffee": _item(live_client, rid, "Filter Coffee", 2500, 20, category="Beverages"),
        "vada": _item(live_client, rid, "Medu Vada", 5000, 1, category="Breakfast"),
    }  # fmt: skip
    _sign_in(live_client, restaurant["slug"])
    return data


def _menu(client: TestClient) -> dict[str, dict]:
    return {i["name"]: i for i in client.get(f"{D}/menu").json()}


def _walk_in(client: TestClient, lines: list[tuple[str, int]], **extra):
    return client.post(f"{D}/orders/walk-in", headers=HDR,
                       json={"items": [{"menu_item_id": i, "quantity": q} for i, q in lines]} | extra)  # fmt: skip


# ── Sign-in, authorization and scoping ───────────────────────────────────────


class TestAuth:
    def test_sign_in_sets_a_safe_cookie_and_never_returns_the_key(self, live_client, shop) -> None:
        live_client.cookies.clear()
        r = live_client.post(f"{D}/session", json={"staff_key": KEY, "restaurant_slug": shop["slug"]})
        assert r.status_code == 200 and r.json()["restaurant_name"] == "Dash Kitchen"
        cookie = r.headers["set-cookie"]
        assert DASHBOARD_COOKIE in cookie and "HttpOnly" in cookie and "SameSite=strict" in cookie
        assert "Path=/api/v1/dashboard" in cookie
        assert KEY not in r.text and KEY not in cookie

    @pytest.mark.parametrize(("method", "path"), [
        ("get", "/summary"), ("get", "/orders"), ("get", "/menu"), ("get", "/session"),
        ("post", "/menu"), ("post", "/orders/walk-in"),
    ])  # fmt: skip
    def test_everything_needs_a_session(self, live_client, shop, method, path) -> None:
        live_client.cookies.clear()
        kwargs = {} if method == "get" else {"json": {}, "headers": HDR}
        assert live_client.request(method.upper(), D + path, **kwargs).status_code == 401

    def test_wrong_key_is_refused_and_throttled(self, live_client, shop) -> None:
        live_client.cookies.clear()
        for _ in range(5):
            assert live_client.post(f"{D}/session", json={"staff_key": "nope"}).status_code == 401
        blocked = live_client.post(f"{D}/session", json={"staff_key": KEY, "restaurant_slug": shop["slug"]})
        assert blocked.status_code == 429

    def test_changes_need_the_dashboard_header(self, live_client, shop) -> None:
        r = live_client.post(f"{D}/menu/{shop['dosa']}/stock", json={"change": 1, "reason": "delivery"})
        assert r.status_code == 403  # a cross-site form cannot send the header

    def test_forged_or_tampered_cookie_is_rejected(self, live_client, shop) -> None:
        token = live_client.cookies.get(DASHBOARD_COOKIE)
        live_client.cookies.set(DASHBOARD_COOKIE, token[:-2] + "xx", path="/api/v1/dashboard")
        assert live_client.get(f"{D}/menu").status_code == 401

    def test_disabled_without_staff_key(self, live_client, shop, monkeypatch) -> None:
        monkeypatch.setattr(settings, "staff_api_key", None)
        assert live_client.get(f"{D}/menu").status_code == 503

    def test_other_restaurants_data_is_out_of_reach(self, live_client, shop) -> None:
        other = _restaurant(live_client, "Other Place")
        foreign_item = _item(live_client, other["id"], "Secret Dish", 9900, 3)
        foreign_order = live_client.post(f"{API}/restaurants/{other['id']}/orders", json={
            "customer_phone": PHONE, "items": [{"menu_item_id": foreign_item, "quantity": 1}]}).json()  # fmt: skip
        assert live_client.get(f"{D}/orders/{foreign_order['id']}").status_code == 404
        assert live_client.post(f"{D}/orders/{foreign_order['id']}/cancel", headers=HDR).status_code == 404
        assert live_client.post(f"{D}/menu/{foreign_item}/stock", headers=HDR,
                                json={"change": 5, "reason": "sneaky"}).status_code == 404  # fmt: skip
        assert "Secret Dish" not in _menu(live_client)
        orders = live_client.get(f"{D}/orders").json()
        assert foreign_order["id"] not in [o["id"] for o in orders["items"]]


# ── Menu & live inventory ────────────────────────────────────────────────────


class TestMenuAndStock:
    def test_create_edit_and_reject_duplicates(self, live_client, shop) -> None:
        r = live_client.post(f"{D}/menu", headers=HDR, json={
            "name": "Rava Idli", "category": "Breakfast", "description": "Semolina idli.",
            "price_paise": 4500, "initial_stock": 12})  # fmt: skip
        assert r.status_code == 201, r.text
        item = r.json()
        assert (item["available_quantity"], item["on_hand"], item["orderable"]) == (12, 12, True)
        dup = live_client.post(f"{D}/menu", headers=HDR, json={"name": "  rava IDLI ", "price_paise": 100})
        assert dup.status_code == 409
        edited = live_client.patch(f"{D}/menu/{item['id']}", headers=HDR, json={"price_paise": 5000})
        assert edited.status_code == 200 and edited.json()["price_paise"] == 5000
        rename = live_client.patch(f"{D}/menu/{item['id']}", headers=HDR, json={"name": "Masala Dosa"})
        assert rename.status_code == 409

    @pytest.mark.parametrize("body", [
        {"name": "Free", "price_paise": 0}, {"name": "Neg", "price_paise": -5},
        {"name": "X", "price_paise": 100, "initial_stock": -1}, {"name": "X", "price_paise": 100, "restaurant_id": str(uuid.uuid4())},
    ])  # fmt: skip
    def test_invalid_items_rejected(self, live_client, shop, body) -> None:
        assert live_client.post(f"{D}/menu", headers=HDR, json=body).status_code == 422

    def test_stock_up_down_and_sold_out(self, live_client, shop, connection) -> None:
        up = live_client.post(f"{D}/menu/{shop['dosa']}/stock", headers=HDR,
                              json={"change": 10, "reason": "Delivery received"}).json()  # fmt: skip
        assert (up["on_hand"], up["available_quantity"]) == (15, 15)
        down = live_client.post(f"{D}/menu/{shop['dosa']}/stock", headers=HDR,
                                json={"change": -15, "reason": "Waste / spoiled: power cut"}).json()  # fmt: skip
        assert (down["on_hand"], down["available_quantity"], down["orderable"]) == (0, 0, False)
        assert live_client.get(f"{D}/summary").json()["out_of_stock"] == 1
        notes = connection.execute(select(InventoryMovement.kind, InventoryMovement.quantity, InventoryMovement.note)
                                   .where(InventoryMovement.menu_item_id == uuid.UUID(shop["dosa"]))
                                   .order_by(InventoryMovement.created_at)).all()  # fmt: skip
        assert [tuple(n) for n in notes[-2:]] == [("restock", 10, "Delivery received"),
                                                  ("adjust", 15, "Waste / spoiled: power cut")]  # fmt: skip

    @pytest.mark.parametrize("body", [
        {"change": 0, "reason": "nothing"}, {"change": 5, "reason": ""}, {"change": 5},
        {"change": 1.5, "reason": "half"}, {"change": "5", "reason": "text"},
    ])  # fmt: skip
    def test_invalid_stock_changes(self, live_client, shop, body) -> None:
        r = live_client.post(f"{D}/menu/{shop['dosa']}/stock", headers=HDR, json=body)
        assert r.status_code in (409, 422)

    def test_cannot_remove_reserved_stock(self, live_client, shop) -> None:
        assert _walk_in(live_client, [(shop["dosa"], 4)]).status_code == 201
        r = live_client.post(f"{D}/menu/{shop['dosa']}/stock", headers=HDR, json={"change": -2, "reason": "count"})
        assert r.status_code == 409 and "reserved" in r.json()["detail"]
        assert _menu(live_client)["Masala Dosa"]["on_hand"] == 5

    def test_switch_off_and_on(self, live_client, shop) -> None:
        off = live_client.patch(f"{D}/menu/{shop['coffee']}", headers=HDR, json={"is_available": False}).json()
        assert off["orderable"] is False
        assert _walk_in(live_client, [(shop["coffee"], 1)]).status_code == 422
        live_client.patch(f"{D}/menu/{shop['coffee']}", headers=HDR, json={"is_available": True})
        assert _walk_in(live_client, [(shop["coffee"], 1)]).status_code == 201

    def test_retiring_keeps_history_and_blocks_new_orders(self, live_client, shop, connection) -> None:
        old = _walk_in(live_client, [(shop["coffee"], 2)]).json()
        live_client.post(f"{D}/orders/{old['id']}/mark-paid", headers=HDR)
        retired = live_client.post(f"{D}/menu/{shop['coffee']}/deactivate", headers=HDR).json()
        assert (retired["is_active"], retired["orderable"]) == (False, False)
        # Gone from the public menu and from new orders...
        public = live_client.get(f"{API}/restaurants/{shop['restaurant']}/menu-items").json()["items"]
        assert "Filter Coffee" not in [i["name"] for i in public]
        r = live_client.post(f"{API}/restaurants/{shop['restaurant']}/orders", json={
            "customer_phone": PHONE, "items": [{"menu_item_id": shop["coffee"], "quantity": 1}]})  # fmt: skip
        assert r.status_code == 422 and "no longer on the menu" in r.json()["detail"][0]["msg"]
        # ...but its past order, lines, payment and ledger are intact.
        history = live_client.get(f"{D}/orders/{old['id']}").json()
        assert history["items"][0]["menu_item_id"] == shop["coffee"]
        assert history["items"][0]["item_name"] == "Filter Coffee" and history["payment"]["status"] == "succeeded"
        assert live_client.get(f"{D}/menu/{shop['coffee']}/movements").json()
        # A new dish may reuse the name; restoring the old one then conflicts.
        assert live_client.post(f"{D}/menu", headers=HDR, json={"name": "Filter Coffee", "price_paise": 3000}).status_code == 201
        assert live_client.post(f"{D}/menu/{shop['coffee']}/activate", headers=HDR).status_code == 409

    def test_cannot_retire_with_unpaid_reservations(self, live_client, shop) -> None:
        _walk_in(live_client, [(shop["dosa"], 1)])
        assert live_client.post(f"{D}/menu/{shop['dosa']}/deactivate", headers=HDR).status_code == 409


# ── Live orders ──────────────────────────────────────────────────────────────


class TestOrders:
    def test_channels_are_labelled_and_filterable(self, live_client, shop) -> None:
        api_order = live_client.post(f"{API}/restaurants/{shop['restaurant']}/orders", json={
            "customer_phone": PHONE, "items": [{"menu_item_id": shop["coffee"], "quantity": 1}]}).json()  # fmt: skip
        walk = _walk_in(live_client, [(shop["coffee"], 2)]).json()
        listed = {o["id"]: o for o in live_client.get(f"{D}/orders").json()["items"]}
        assert (listed[api_order["id"]]["source"], listed[walk["id"]]["source"]) == ("API", "WALK_IN_POS")
        only_walk = live_client.get(f"{D}/orders", params={"source": "WALK_IN_POS"}).json()["items"]
        assert [o["id"] for o in only_walk] == [walk["id"]]

    def test_walk_in_without_contact_details(self, live_client, shop) -> None:
        r = _walk_in(live_client, [(shop["dosa"], 2), (shop["coffee"], 1)], notes="Table 4")
        assert r.status_code == 201, r.text
        order = r.json()
        assert order["source"] == "WALK_IN_POS" and order["customer_name"] is None and order["customer_phone"] is None
        assert order["total_paise"] == 2 * 8000 + 2500  # prices from the database
        assert order["inventory_state"] == "reserved" and order["payment"]["status"] == "pending"

    def test_client_cannot_set_prices_source_or_status(self, live_client, shop) -> None:
        for extra in ({"total_paise": 1}, {"source": "AI_VOICE"}, {"status": "completed"}, {"restaurant_id": str(uuid.uuid4())}):
            assert _walk_in(live_client, [(shop["dosa"], 1)], **extra).status_code == 422

    def test_payment_and_preparation_flow(self, live_client, shop) -> None:
        order = _walk_in(live_client, [(shop["dosa"], 2)]).json()
        assert live_client.post(f"{D}/orders/{order['id']}/status", headers=HDR,
                                json={"status": "confirmed"}).status_code == 409  # only via payment  # fmt: skip
        paid = live_client.post(f"{D}/orders/{order['id']}/mark-paid", headers=HDR).json()
        assert (paid["status"], paid["payment"]["status"], paid["inventory_state"]) == ("confirmed", "succeeded", "sold")
        assert live_client.post(f"{D}/orders/{order['id']}/mark-paid", headers=HDR).status_code == 409
        for step in ("preparing", "ready", "completed"):
            r = live_client.post(f"{D}/orders/{order['id']}/status", headers=HDR, json={"status": step})
            assert r.status_code == 200 and r.json()["status"] == step
        assert _menu(live_client)["Masala Dosa"]["sold"] == 2

    def test_search_and_active_filter(self, live_client, shop) -> None:
        a = _walk_in(live_client, [(shop["coffee"], 1)], customer_name="Asha Rao", customer_phone="+91 90000 12345").json()
        b = _walk_in(live_client, [(shop["coffee"], 1)]).json()
        live_client.post(f"{D}/orders/{b['id']}/cancel", headers=HDR)
        by_name = live_client.get(f"{D}/orders", params={"q": "asha"}).json()["items"]
        by_phone = live_client.get(f"{D}/orders", params={"q": "12345"}).json()["items"]
        by_number = live_client.get(f"{D}/orders", params={"q": a["order_number"]}).json()["items"]
        assert [o["id"] for o in by_name] == [o["id"] for o in by_phone] == [o["id"] for o in by_number] == [a["id"]]
        active = live_client.get(f"{D}/orders", params={"active_only": True}).json()["items"]
        assert a["id"] in [o["id"] for o in active] and b["id"] not in [o["id"] for o in active]

    def test_summary_counts(self, live_client, shop) -> None:
        _walk_in(live_client, [(shop["vada"], 1)])  # takes the last vada (reserved)
        s = live_client.get(f"{D}/summary").json()
        assert (s["active_orders"], s["orders_today"], s["awaiting_payment"]) == (1, 1, 1)
        assert s["out_of_stock"] == 1 and s["low_stock"] == 1  # vada sold out; dosa (5) is low

    def test_expiry_releases_stock_and_shows_on_dashboard(self, live_client, shop, connection) -> None:
        order = _walk_in(live_client, [(shop["vada"], 1)]).json()
        assert _menu(live_client)["Medu Vada"]["available_quantity"] == 0
        connection.execute(update(Order).where(Order.id == uuid.UUID(order["id"]))
                           .values(reservation_expires_at=datetime.now(UTC) - timedelta(seconds=1)))  # fmt: skip
        live_client.post(f"{API}/maintenance/expire-reservations")
        after = live_client.get(f"{D}/orders/{order['id']}").json()
        assert (after["status"], after["cancel_reason"], after["payment"]["status"]) == ("cancelled", "expired", "expired")
        assert _menu(live_client)["Medu Vada"]["available_quantity"] == 1

    def test_change_feed_moves_on_every_change(self, live_client, shop, session) -> None:
        rid = uuid.UUID(shop["restaurant"])
        before = dashboard_service.change_fingerprint(session, rid)
        live_client.post(f"{D}/menu/{shop['coffee']}/stock", headers=HDR, json={"change": 1, "reason": "count"})
        after_stock = dashboard_service.change_fingerprint(session, rid)
        _walk_in(live_client, [(shop["coffee"], 1)])
        after_order = dashboard_service.change_fingerprint(session, rid)
        assert len({before, after_stock, after_order}) == 3


# ── The agent sees dashboard changes on its next turn ────────────────────────


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


class TestAgentSync:
    def test_stock_price_and_retirement_reach_the_agent(self, agent, shop, live_client) -> None:
        live_client.post(f"{D}/menu/{shop['dosa']}/stock", headers=HDR, json={"change": -5, "reason": "count"})
        live_client.patch(f"{D}/menu/{shop['coffee']}", headers=HDR, json={"price_paise": 3000})
        live_client.post(f"{D}/menu/{shop['vada']}/deactivate", headers=HDR)
        provider = ScriptedProvider(call("get_menu", available_only=False),
                                    say("Masala dosa is sold out; filter coffee is INR 30.00."))  # fmt: skip
        agent(provider)
        live_client.post(f"{API}/restaurants/{shop['restaurant']}/agent/chat", json={"message": "What do you have?"})
        items = {i["name"]: i for i in last_tool_result(provider.requests[1]["messages"])["items"]}
        assert (items["Masala Dosa"]["available_quantity"], items["Masala Dosa"]["orderable"]) == (0, False)
        assert items["Filter Coffee"]["price"] == "INR 30.00"
        assert "Medu Vada" not in items  # retired dishes are not offered

    def test_agent_order_is_refused_after_a_counter_sale_takes_the_last_item(self, agent, shop, live_client) -> None:
        provider = ScriptedProvider(
            call("create_order", customer_phone=PHONE, items=[{"menu_item_id": shop["vada"], "quantity": 1}]),
            say("One medu vada, INR 50.00. Shall I place it?"),
        )  # fmt: skip
        client = agent(provider)
        first = client.post(f"{API}/restaurants/{shop['restaurant']}/agent/chat", json={"message": "One vada"}).json()
        assert _walk_in(live_client, [(shop["vada"], 1)]).status_code == 201  # counter sale wins the last one
        placed = client.post(f"{API}/restaurants/{shop['restaurant']}/agent/chat", json={
            "session_id": first["session_id"], "confirm_proposal_id": first["pending_order"]["proposal_id"]}).json()  # fmt: skip
        assert placed["order"] is None and "only 0 of 'Medu Vada' available" in placed["reply"]
        sources = {o["source"] for o in live_client.get(f"{D}/orders").json()["items"]}
        assert sources == {"WALK_IN_POS"}

    def test_agent_orders_are_labelled_ai_chat(self, agent, shop, live_client) -> None:
        provider = ScriptedProvider(
            call("create_order", customer_phone=PHONE, items=[{"menu_item_id": shop["coffee"], "quantity": 1}]),
            say("One filter coffee. Shall I place it?"),
        )  # fmt: skip
        client = agent(provider)
        first = client.post(f"{API}/restaurants/{shop['restaurant']}/agent/chat", json={"message": "A coffee"}).json()
        placed = client.post(f"{API}/restaurants/{shop['restaurant']}/agent/chat", json={
            "session_id": first["session_id"], "confirm_proposal_id": first["pending_order"]["proposal_id"]}).json()  # fmt: skip
        assert placed["order"]["source"] == "AI_CHAT"
        assert live_client.get(f"{D}/orders/{placed['order']['id']}").json()["source"] == "AI_CHAT"
        assert last_tool_result(provider.requests[1]["messages"])["status"] == "awaiting_customer_confirmation"
