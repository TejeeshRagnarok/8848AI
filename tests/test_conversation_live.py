"""Conversational behaviour backed by real PostgreSQL data (``live_db``):
the per-turn menu/cart snapshot, most-ordered answers from paid history,
multi-turn cart changes and the contact-number requirement.  The LLM is the
scripted fake; every test runs in the rolled-back session from conftest.
"""

import uuid
from collections.abc import Generator

import pytest
from agent_fakes import ScriptedProvider, call, last_tool_result, say
from db_helpers import count_test_orders
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.agent.sessions import InMemorySessionStore
from app.main import app
from app.models import InventoryItem
from app.routers.agent import get_provider_factory, get_session_store
from app.services.insights import MIN_ORDERS_FOR_POPULARITY

pytestmark = pytest.mark.live_db

API = "/api/v1"
PHONE = "+91 90000 00001"


def _restaurant(client: TestClient) -> str:
    r = client.post(f"{API}/restaurants", json={"name": "Chat Kitchen", "slug": f"test-{uuid.uuid4().hex[:12]}"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _item(client: TestClient, rid: str, name: str, price: int, stock: int | None = None, **extra) -> str:
    r = client.post(f"{API}/restaurants/{rid}/menu-items", json={"name": name, "price_paise": price} | extra)
    assert r.status_code == 201, r.text
    if stock is not None:
        assert client.put(f"{API}/menu-items/{r.json()['id']}/inventory", json={"on_hand": stock}).status_code == 200
    return r.json()["id"]


@pytest.fixture
def menu(live_client: TestClient) -> dict:
    rid = _restaurant(live_client)
    return {
        "restaurant": rid,
        "dosa": _item(live_client, rid, "Masala Dosa", 8000, 5, category="Dosa",
                      description="Crisp dosa filled with spiced potato masala."),
        "plain": _item(live_client, rid, "Plain Dosa", 6000, 10, category="Dosa"),
        "coffee": _item(live_client, rid, "Filter Coffee", 2500, 20, category="Beverages"),
        "vada": _item(live_client, rid, "Medu Vada", 5000, 10, category="Breakfast"),
        "kesari": _item(live_client, rid, "Kesari", 4000, 0, category="Sweets"),
        "thali": _item(live_client, rid, "Special Thali", 20000, is_available=False),
    }  # fmt: skip


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


def _chat(client: TestClient, rid: str, **body) -> dict:
    r = client.post(f"{API}/restaurants/{rid}/agent/chat", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _context(provider: ScriptedProvider, request: int = 0) -> str:
    return provider.requests[request]["messages"][1]["content"]


def _paid_order(client: TestClient, rid: str, lines: list[tuple[str, int]]) -> dict:
    order = client.post(f"{API}/restaurants/{rid}/orders", json={
        "customer_phone": PHONE, "items": [{"menu_item_id": i, "quantity": q} for i, q in lines]}).json()  # fmt: skip
    pay = client.post(f"{API}/payments/{order['payment']['id']}/simulate",
                      json={"outcome": "succeeded", "amount_paise": order["total_paise"]})  # fmt: skip
    assert pay.status_code == 200, pay.text
    return order


# ── Live menu and cart snapshot ──────────────────────────────────────────────


class TestLiveContext:
    def test_snapshot_has_real_prices_stock_and_empty_cart(self, agent, menu) -> None:
        provider = ScriptedProvider(say("We have dosas, coffee and vada."))
        _chat(agent(provider), menu["restaurant"], message="What's on the menu?")
        context = _context(provider)
        assert context.startswith("Live restaurant data")
        assert f'"Masala Dosa" | {menu["dosa"]} | "Dosa" | INR 80.00 | available (5 left)' in context
        assert '"Kesari"' in context and "sold out" in context
        assert '"Special Thali"' in context and "not available now" in context
        assert "Current cart: empty." in context
        assert "potato masala" not in context  # descriptions stay in tool results (data)

    def test_price_change_shows_on_the_next_turn(self, agent, menu, live_client) -> None:
        provider = ScriptedProvider(say("Masala dosa is INR 80.00."), say("It is now INR 90.00."))
        client = agent(provider)
        first = _chat(client, menu["restaurant"], message="How much is masala dosa?")
        live_client.patch(f"{API}/menu-items/{menu['dosa']}", json={"price_paise": 9000})
        _chat(client, menu["restaurant"], session_id=first["session_id"], message="And now?")
        assert "INR 90.00" in _context(provider, 1) and "INR 80.00" not in _context(provider, 1)


# ── Most ordered ─────────────────────────────────────────────────────────────


class TestPopularity:
    def test_too_little_history_is_reported_honestly(self, agent, menu, live_client) -> None:
        _paid_order(live_client, menu["restaurant"], [(menu["dosa"], 1)])
        provider = ScriptedProvider(call("get_popular_items"),
                                    say("We don't have enough order history yet; the masala dosa is a good pick."))  # fmt: skip
        _chat(agent(provider), menu["restaurant"], message="What's your most ordered dish?")
        result = last_tool_result(provider.requests[1]["messages"])
        assert result["status"] == "insufficient_data"
        assert result["paid_orders"] == 1 and result["minimum_needed"] == MIN_ORDERS_FOR_POPULARITY
        assert "most_ordered" not in result

    def test_ranking_uses_paid_orders_only(self, agent, menu, live_client) -> None:
        for _ in range(MIN_ORDERS_FOR_POPULARITY):
            _paid_order(live_client, menu["restaurant"], [(menu["coffee"], 2), (menu["dosa"], 1)])
        # Unpaid and cancelled orders must not count, however large.
        live_client.post(f"{API}/restaurants/{menu['restaurant']}/orders", json={
            "customer_phone": PHONE, "items": [{"menu_item_id": menu["plain"], "quantity": 9}]})  # fmt: skip
        cancelled = live_client.post(f"{API}/restaurants/{menu['restaurant']}/orders", json={
            "customer_phone": PHONE, "items": [{"menu_item_id": menu["vada"], "quantity": 9}]}).json()  # fmt: skip
        live_client.patch(f"{API}/orders/{cancelled['id']}/status", json={"status": "cancelled"})

        provider = ScriptedProvider(call("get_popular_items"), say("Filter coffee is our most ordered."))
        _chat(agent(provider), menu["restaurant"], message="What's popular here?")
        result = last_tool_result(provider.requests[1]["messages"])
        assert result["status"] == "ok" and result["paid_orders"] == MIN_ORDERS_FOR_POPULARITY
        assert [(i["name"], i["units_sold"], i["in_orders"]) for i in result["most_ordered"]] == [
            ("Filter Coffee", 10, 5), ("Masala Dosa", 5, 5),
        ]  # fmt: skip
        # Those 5 sales used up all 5 masala dosas: popular, but not orderable now.
        assert {i["name"]: i["available_now"] for i in result["most_ordered"]} == {
            "Filter Coffee": True, "Masala Dosa": False,
        }  # fmt: skip


# ── Cart across turns ────────────────────────────────────────────────────────


class TestCartConversation:
    def test_add_remove_and_total_across_turns(self, agent, menu, connection) -> None:
        provider = ScriptedProvider(
            call("create_order", items=[{"menu_item_id": menu["dosa"], "quantity": 2},
                                        {"menu_item_id": menu["coffee"], "quantity": 1}]),
            say("Two masala dosas and a filter coffee come to INR 185.00. Anything else?"),
            call("create_order", items=[{"menu_item_id": menu["dosa"], "quantity": 2},
                                        {"menu_item_id": menu["vada"], "quantity": 1}]),
            say("Done: coffee removed, one medu vada added. The total is INR 210.00."),
            say("Your total is INR 210.00."),
        )  # fmt: skip
        client = agent(provider)
        first = _chat(client, menu["restaurant"], message="Two masala dosas and a coffee")
        sid = first["session_id"]
        second = _chat(client, menu["restaurant"], session_id=sid, message="Remove the coffee and add a vada")
        _chat(client, menu["restaurant"], session_id=sid, message="What is my total now?")

        assert first["pending_order"]["total_paise"] == 18500
        assert second["pending_order"]["total_paise"] == 21000
        cart = _context(provider, 4)  # the "total" turn sees the server's cart
        assert 'Current cart (not placed yet): 2 x "Masala Dosa" (INR 160.00), 1 x "Medu Vada" (INR 50.00)' in cart
        assert "total INR 210.00" in cart and "contact number not given yet" in cart
        # History is what was said, not tool output.
        roles = [m["role"] for m in provider.requests[4]["messages"][2:]]
        assert roles == ["user", "assistant", "user", "assistant", "user"]
        assert count_test_orders(connection) == 0
        assert connection.execute(select(InventoryItem.reserved).where(
            InventoryItem.menu_item_id == uuid.UUID(menu["dosa"]))).scalar_one() == 0  # fmt: skip

    def test_contact_number_required_before_placing(self, agent, menu, connection) -> None:
        provider = ScriptedProvider(
            call("create_order", items=[{"menu_item_id": menu["dosa"], "quantity": 1}]),
            say("One masala dosa, INR 80.00. Shall I place it?"),
            call("create_order", customer_phone=PHONE, items=[{"menu_item_id": menu["dosa"], "quantity": 1}]),
            say("Thanks. One masala dosa, INR 80.00, on that number. Shall I place it?"),
        )  # fmt: skip
        client = agent(provider)
        first = _chat(client, menu["restaurant"], message="One masala dosa")
        sid, proposal = first["session_id"], first["pending_order"]["proposal_id"]
        assert first["pending_order"]["customer_phone"] is None

        asked = _chat(client, menu["restaurant"], session_id=sid, message="Yes", confirm_proposal_id=proposal)
        assert asked["order"] is None and "phone number" in asked["reply"]
        assert asked["pending_order"]["proposal_id"] == proposal  # cart kept
        assert count_test_orders(connection) == 0

        second = _chat(client, menu["restaurant"], session_id=sid, message=PHONE)
        placed = _chat(client, menu["restaurant"], session_id=sid, message="Yes",
                       confirm_proposal_id=second["pending_order"]["proposal_id"])  # fmt: skip
        assert placed["order"]["customer_phone"] == PHONE and placed["order"]["status"] == "pending"
        assert placed["order"]["inventory_state"] == "reserved"
        assert count_test_orders(connection) == 1
