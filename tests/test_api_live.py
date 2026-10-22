"""Live API tests: real HTTP requests through FastAPI into real PostgreSQL.

Marked ``live_db``.  Requests use the ``live_client`` fixture, whose database
session lives inside a transaction that is rolled back after each test, so
no restaurants or menu items are left behind.
"""

import uuid
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.live_db

API = "/api/v1"


def _slug() -> str:
    return f"test-{uuid.uuid4().hex[:12]}"


def _create_restaurant(client: TestClient, **fields) -> dict:
    response = client.post(f"{API}/restaurants", json={"name": "Test Kitchen", "slug": _slug()} | fields)
    assert response.status_code == 201, response.text
    return response.json()


def _create_item(client: TestClient, restaurant_id: str, **fields) -> dict:
    body = {"name": "Masala Dosa", "price_paise": 12000} | fields
    response = client.post(f"{API}/restaurants/{restaurant_id}/menu-items", json=body)
    assert response.status_code == 201, response.text
    return response.json()


class TestRestaurantApi:
    def test_create_and_retrieve(self, live_client: TestClient) -> None:
        created = _create_restaurant(live_client, name="Spice Garden", phone="+91 98765 43210")
        assert uuid.UUID(created["id"])
        assert created["timezone"] == "Asia/Kolkata"
        assert created["currency"] == "INR"
        assert created["is_active"] is True
        assert created["created_at"] and created["updated_at"]

        fetched = live_client.get(f"{API}/restaurants/{created['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == created

    def test_list_and_paginate(self, live_client: TestClient) -> None:
        before = live_client.get(f"{API}/restaurants").json()["total"]
        mine = [_create_restaurant(live_client)["id"] for _ in range(3)]

        page1 = live_client.get(f"{API}/restaurants", params={"limit": 2, "offset": before}).json()
        page2 = live_client.get(f"{API}/restaurants", params={"limit": 2, "offset": before + 2}).json()
        assert page1["total"] == before + 3
        assert (page1["limit"], page1["offset"]) == (2, before)
        assert [r["id"] for r in page1["items"]] + [r["id"] for r in page2["items"]] == mine

    def test_partial_update(self, live_client: TestClient) -> None:
        created = _create_restaurant(live_client, phone="+91 11111 11111")
        response = live_client.patch(
            f"{API}/restaurants/{created['id']}", json={"name": "Renamed", "phone": None}
        )
        assert response.status_code == 200
        updated = response.json()
        assert updated["name"] == "Renamed"
        assert updated["phone"] is None
        assert updated["slug"] == created["slug"]  # untouched
        assert updated["id"] == created["id"]
        assert datetime.fromisoformat(updated["updated_at"]) > datetime.fromisoformat(
            created["updated_at"]
        )

    def test_duplicate_slug_on_create_returns_409(self, live_client: TestClient) -> None:
        existing = _create_restaurant(live_client)
        response = live_client.post(
            f"{API}/restaurants", json={"name": "Copycat", "slug": existing["slug"]}
        )
        assert response.status_code == 409
        assert response.json() == {"detail": "A restaurant with this slug already exists."}
        # The failed transaction was rolled back; the session is still usable.
        assert live_client.get(f"{API}/restaurants/{existing['id']}").status_code == 200

    def test_duplicate_slug_on_update_returns_409(self, live_client: TestClient) -> None:
        first = _create_restaurant(live_client)
        second = _create_restaurant(live_client)
        response = live_client.patch(
            f"{API}/restaurants/{second['id']}", json={"slug": first["slug"]}
        )
        assert response.status_code == 409
        assert live_client.get(f"{API}/restaurants/{second['id']}").json()["slug"] == second["slug"]

    def test_missing_restaurant_returns_404(self, live_client: TestClient) -> None:
        missing = uuid.uuid4()
        assert live_client.get(f"{API}/restaurants/{missing}").status_code == 404
        assert live_client.patch(f"{API}/restaurants/{missing}", json={"name": "X"}).status_code == 404
        assert live_client.delete(f"{API}/restaurants/{missing}").status_code == 404

    def test_delete_deactivates_instead_of_removing(self, live_client: TestClient) -> None:
        created = _create_restaurant(live_client)
        _create_item(live_client, created["id"])  # a hard delete would be blocked by RESTRICT

        response = live_client.delete(f"{API}/restaurants/{created['id']}")
        assert response.status_code == 200
        assert response.json()["is_active"] is False

        # Still retrievable, clearly marked inactive, and its menu is preserved.
        fetched = live_client.get(f"{API}/restaurants/{created['id']}").json()
        assert fetched["is_active"] is False
        menu = live_client.get(f"{API}/restaurants/{created['id']}/menu-items").json()
        assert menu["total"] == 1

        # Repeating the delete is harmless.
        assert live_client.delete(f"{API}/restaurants/{created['id']}").status_code == 200

    def test_inactive_restaurants_hidden_from_default_list(self, live_client: TestClient) -> None:
        active = _create_restaurant(live_client)
        inactive = _create_restaurant(live_client)
        live_client.delete(f"{API}/restaurants/{inactive['id']}")

        default_ids = {
            r["id"] for r in live_client.get(f"{API}/restaurants", params={"limit": 100}).json()["items"]
        }
        assert active["id"] in default_ids
        assert inactive["id"] not in default_ids

        all_page = live_client.get(
            f"{API}/restaurants", params={"include_inactive": True, "limit": 100}
        ).json()
        listed = {r["id"]: r["is_active"] for r in all_page["items"]}
        assert listed[inactive["id"]] is False
        assert listed[active["id"]] is True

    def test_explicit_reactivation(self, live_client: TestClient) -> None:
        created = _create_restaurant(live_client)
        live_client.delete(f"{API}/restaurants/{created['id']}")
        response = live_client.patch(f"{API}/restaurants/{created['id']}", json={"is_active": True})
        assert response.json()["is_active"] is True


class TestMenuItemApi:
    def test_create_for_existing_restaurant(self, live_client: TestClient) -> None:
        restaurant = _create_restaurant(live_client)
        item = _create_item(
            live_client, restaurant["id"], description="Crispy", category="South Indian"
        )
        assert item["restaurant_id"] == restaurant["id"]
        assert item["price_paise"] == 12000
        assert item["is_available"] is True

        fetched = live_client.get(f"{API}/menu-items/{item['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == item

    def test_rejected_for_inactive_restaurant(self, live_client: TestClient) -> None:
        restaurant = _create_restaurant(live_client)
        live_client.delete(f"{API}/restaurants/{restaurant['id']}")
        response = live_client.post(
            f"{API}/restaurants/{restaurant['id']}/menu-items",
            json={"name": "Dosa", "price_paise": 100},
        )
        assert response.status_code == 409
        assert "inactive" in response.json()["detail"]
        assert live_client.get(f"{API}/restaurants/{restaurant['id']}/menu-items").json()["total"] == 0

    def test_rejected_for_missing_restaurant(self, live_client: TestClient) -> None:
        response = live_client.post(
            f"{API}/restaurants/{uuid.uuid4()}/menu-items",
            json={"name": "Dosa", "price_paise": 100},
        )
        assert response.status_code == 404
        assert live_client.get(f"{API}/restaurants/{uuid.uuid4()}/menu-items").status_code == 404

    @pytest.mark.parametrize("price", [0, -1])
    def test_non_positive_price_rejected(self, live_client: TestClient, price: int) -> None:
        restaurant = _create_restaurant(live_client)
        response = live_client.post(
            f"{API}/restaurants/{restaurant['id']}/menu-items",
            json={"name": "Free Lunch", "price_paise": price},
        )
        assert response.status_code == 422
        item = _create_item(live_client, restaurant["id"])
        response = live_client.patch(f"{API}/menu-items/{item['id']}", json={"price_paise": price})
        assert response.status_code == 422
        assert live_client.get(f"{API}/menu-items/{item['id']}").json()["price_paise"] == 12000

    def test_list_is_scoped_paginated_and_filterable(self, live_client: TestClient) -> None:
        restaurant = _create_restaurant(live_client)
        other = _create_restaurant(live_client)
        _create_item(live_client, restaurant["id"], name="Idli")
        _create_item(live_client, restaurant["id"], name="Vada", is_available=False)
        _create_item(live_client, restaurant["id"], name="Dosa")
        _create_item(live_client, other["id"], name="Elsewhere")

        url = f"{API}/restaurants/{restaurant['id']}/menu-items"
        everything = live_client.get(url).json()
        assert everything["total"] == 3
        assert [i["name"] for i in everything["items"]] == ["Dosa", "Idli", "Vada"]
        assert all(i["restaurant_id"] == restaurant["id"] for i in everything["items"])

        page = live_client.get(url, params={"limit": 2, "offset": 2}).json()
        assert [i["name"] for i in page["items"]] == ["Vada"]

        available = live_client.get(url, params={"is_available": True}).json()
        assert [i["name"] for i in available["items"]] == ["Dosa", "Idli"]
        unavailable = live_client.get(url, params={"is_available": False}).json()
        assert [i["name"] for i in unavailable["items"]] == ["Vada"]

    def test_update_price_and_availability(self, live_client: TestClient) -> None:
        restaurant = _create_restaurant(live_client)
        item = _create_item(live_client, restaurant["id"], category="Breakfast")
        response = live_client.patch(
            f"{API}/menu-items/{item['id']}", json={"price_paise": 15050, "is_available": False}
        )
        assert response.status_code == 200
        updated = response.json()
        assert updated["price_paise"] == 15050
        assert updated["is_available"] is False
        assert updated["name"] == item["name"]
        assert updated["category"] == "Breakfast"
        assert updated["restaurant_id"] == restaurant["id"]

    def test_missing_menu_item_returns_404(self, live_client: TestClient) -> None:
        missing = uuid.uuid4()
        assert live_client.get(f"{API}/menu-items/{missing}").status_code == 404
        assert live_client.patch(f"{API}/menu-items/{missing}", json={"name": "X"}).status_code == 404
        assert live_client.delete(f"{API}/menu-items/{missing}").status_code == 404

    def test_delete_menu_item(self, live_client: TestClient) -> None:
        restaurant = _create_restaurant(live_client)
        keep = _create_item(live_client, restaurant["id"], name="Keep")
        gone = _create_item(live_client, restaurant["id"], name="Gone")

        response = live_client.delete(f"{API}/menu-items/{gone['id']}")
        assert response.status_code == 204
        assert response.content == b""
        assert live_client.get(f"{API}/menu-items/{gone['id']}").status_code == 404
        remaining = live_client.get(f"{API}/restaurants/{restaurant['id']}/menu-items").json()
        assert [i["id"] for i in remaining["items"]] == [keep["id"]]


class TestHealthLive:
    def test_health_endpoints(self, live_client: TestClient) -> None:
        assert live_client.get("/health").status_code == 200
        response = live_client.get("/health/db")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "database": "connected"}
