"""API tests with a mocked database session — no PostgreSQL required.

These cover request validation, error mapping and leak-safety.  Successful
CRUD flows against real PostgreSQL are in ``test_api_live.py``.
"""

import uuid
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.database import get_db
from app.main import app

SECRET = "s3cr3t-password"
LEAKY_SQL = f"INSERT INTO restaurants ... postgresql+psycopg://app:{SECRET}@db/x"
RID = uuid.uuid4()
VALID_RESTAURANT = {"name": "Spice Garden", "slug": "spice-garden"}


@pytest.fixture
def db() -> Generator[MagicMock, None, None]:
    session = MagicMock(spec=Session)

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield session
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def client(db: MagicMock) -> TestClient:
    return TestClient(app)


def _integrity_error(constraint: str | None) -> IntegrityError:
    orig = Exception(f"duplicate key value violates {constraint}; {LEAKY_SQL}")
    orig.diag = MagicMock(constraint_name=constraint)  # type: ignore[attr-defined]
    return IntegrityError(LEAKY_SQL, {"password": SECRET}, orig)


def _assert_no_leak(text: str) -> None:
    for fragment in (SECRET, "INSERT", "psycopg", "Traceback", "duplicate key"):
        assert fragment not in text


class TestIntegrityErrorHandling:
    def test_duplicate_slug_returns_409_and_rolls_back(self, client, db) -> None:
        db.commit.side_effect = _integrity_error("uq_restaurants_slug")
        response = client.post("/api/v1/restaurants", json=VALID_RESTAURANT)
        assert response.status_code == 409
        assert response.json() == {"detail": "A restaurant with this slug already exists."}
        db.rollback.assert_called_once()
        _assert_no_leak(response.text)

    def test_unknown_constraint_gets_generic_message(self, client, db) -> None:
        db.commit.side_effect = _integrity_error("some_internal_constraint")
        response = client.post("/api/v1/restaurants", json=VALID_RESTAURANT)
        assert response.status_code == 409
        assert response.json() == {"detail": "The request conflicts with existing data."}
        _assert_no_leak(response.text)

    def test_integrity_error_outside_commit_is_caught_by_safety_net(self, client, db) -> None:
        db.get.side_effect = _integrity_error(None)
        response = client.get(f"/api/v1/restaurants/{RID}")
        assert response.status_code == 409
        _assert_no_leak(response.text)


class TestDatabaseFailures:
    def test_operational_error_returns_503_without_details(self, client, db) -> None:
        db.get.side_effect = OperationalError(LEAKY_SQL, {}, Exception(SECRET))
        response = client.get(f"/api/v1/restaurants/{RID}")
        assert response.status_code == 503
        assert response.json() == {"status": "error", "database": "unreachable"}
        _assert_no_leak(response.text)


class TestNotFound:
    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("get", f"/api/v1/restaurants/{RID}", None),
            ("patch", f"/api/v1/restaurants/{RID}", {"name": "X"}),
            ("delete", f"/api/v1/restaurants/{RID}", None),
            ("get", f"/api/v1/restaurants/{RID}/menu-items", None),
            ("post", f"/api/v1/restaurants/{RID}/menu-items", {"name": "X", "price_paise": 1}),
            ("get", f"/api/v1/menu-items/{RID}", None),
            ("patch", f"/api/v1/menu-items/{RID}", {"price_paise": 1}),
            ("delete", f"/api/v1/menu-items/{RID}", None),
        ],
    )
    def test_missing_resources_return_404(self, client, db, method, path, body) -> None:
        db.get.return_value = None
        response = client.request(method, path, json=body)
        assert response.status_code == 404
        assert response.json()["detail"].endswith("not found.")
        db.commit.assert_not_called()


class TestRequestValidation:
    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("post", "/api/v1/restaurants", {"name": "X"}),
            ("post", "/api/v1/restaurants", VALID_RESTAURANT | {"id": str(RID)}),
            ("post", "/api/v1/restaurants", VALID_RESTAURANT | {"slug": "Not A Slug"}),
            ("patch", f"/api/v1/restaurants/{RID}", {"id": str(uuid.uuid4())}),
            ("patch", f"/api/v1/restaurants/{RID}", {"name": None}),
            ("post", f"/api/v1/restaurants/{RID}/menu-items", {"name": "X", "price_paise": 0}),
            ("post", f"/api/v1/restaurants/{RID}/menu-items", {"name": "X", "price_paise": -100}),
            ("post", f"/api/v1/restaurants/{RID}/menu-items", {"name": "X", "price_paise": 99.5}),
            ("patch", f"/api/v1/menu-items/{RID}", {"price_paise": 0}),
            ("patch", f"/api/v1/menu-items/{RID}", {"restaurant_id": str(uuid.uuid4())}),
            ("get", "/api/v1/restaurants/not-a-uuid", None),
            ("get", "/api/v1/restaurants?limit=0", None),
            ("get", "/api/v1/restaurants?limit=101", None),
            ("get", "/api/v1/restaurants?offset=-1", None),
        ],
    )
    def test_invalid_requests_return_422_without_touching_db(
        self, client, db, method, path, body
    ) -> None:
        response = client.request(method, path, json=body)
        assert response.status_code == 422
        db.add.assert_not_called()
        db.commit.assert_not_called()


class TestHealthUnaffected:
    def test_health_still_ok(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_db_uses_session(self, client, db) -> None:
        response = client.get("/health/db")
        assert response.status_code == 200
        db.execute.assert_called_once()
