"""Order API and service tests with mocks — no PostgreSQL required.

Covers HTTP status mapping, request validation, the concurrent-duplicate
(unique-violation) code path, rollback on failure, and leak safety.  Real
database behaviour is in ``test_orders_live.py`` and
``test_orders_concurrency_live.py``.
"""

import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.main import app
from app.schemas import OrderCreate
from app.services import orders as order_service
from app.services.common import ConflictError, UnprocessableError
from app.services.orders import OrderCreationResult, request_fingerprint

RID = uuid.uuid4()
ITEM = uuid.uuid4()
PHONE = "+91 98765 43210"
SECRET = "s3cr3t-password"
BODY = {
    "customer_name": "Asha",
    "customer_phone": PHONE,
    "items": [{"menu_item_id": str(ITEM), "quantity": 2}],
    "idempotency_key": "call-123",
}
NOW = datetime(2026, 10, 22, tzinfo=UTC)


def _fake_order(**overrides) -> SimpleNamespace:
    data = dict(
        id=uuid.uuid4(), restaurant_id=RID, order_number="ORD-100001", customer_name="Asha",
        customer_phone=PHONE, status="pending", subtotal_paise=24000, total_paise=24000,
        notes=None, idempotency_key="call-123", created_at=NOW, updated_at=NOW,
        request_hash=request_fingerprint(OrderCreate(**BODY)),
        items=[SimpleNamespace(
            id=uuid.uuid4(), menu_item_id=ITEM, item_name="Masala Dosa", unit_price_paise=12000,
            quantity=2, line_total_paise=24000, created_at=NOW,
        )],
    )  # fmt: skip
    return SimpleNamespace(**(data | overrides))


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


def _assert_no_leak(text: str) -> None:
    for fragment in (SECRET, PHONE, "INSERT", "psycopg", "Traceback"):
        assert fragment not in text


# ── HTTP mapping ─────────────────────────────────────────────────────────────


class TestCreateResponses:
    def test_new_order_returns_201(self, client, monkeypatch) -> None:
        monkeypatch.setattr(
            order_service, "create_order", lambda *a: OrderCreationResult(_fake_order(), True)
        )
        response = client.post(f"/api/v1/restaurants/{RID}/orders", json=BODY)
        assert response.status_code == 201
        assert "Idempotent-Replayed" not in response.headers
        assert response.json()["total_paise"] == 24000

    def test_idempotent_replay_returns_200_with_header(self, client, monkeypatch) -> None:
        monkeypatch.setattr(
            order_service, "create_order", lambda *a: OrderCreationResult(_fake_order(), False)
        )
        response = client.post(f"/api/v1/restaurants/{RID}/orders", json=BODY)
        assert response.status_code == 200
        assert response.headers["Idempotent-Replayed"] == "true"

    def test_conflict_returns_409(self, client, monkeypatch) -> None:
        def raise_conflict(*a):
            raise ConflictError("idempotency_key was already used for a different order request.")

        monkeypatch.setattr(order_service, "create_order", raise_conflict)
        response = client.post(f"/api/v1/restaurants/{RID}/orders", json=BODY)
        assert response.status_code == 409
        assert "idempotency_key" in response.json()["detail"]

    def test_unprocessable_items_return_422_in_fastapi_format(self, client, monkeypatch) -> None:
        detail = [{"type": "menu_item_unavailable", "loc": ["body", "items", 0, "menu_item_id"],
                   "msg": "'Masala Dosa' is currently unavailable.", "input": str(ITEM)}]

        def raise_unprocessable(*a):
            raise UnprocessableError(detail)

        monkeypatch.setattr(order_service, "create_order", raise_unprocessable)
        response = client.post(f"/api/v1/restaurants/{RID}/orders", json=BODY)
        assert response.status_code == 422
        assert response.json() == {"detail": detail}


class TestRequestValidation:
    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("post", f"/api/v1/restaurants/{RID}/orders", BODY | {"items": []}),
            ("post", f"/api/v1/restaurants/{RID}/orders",
             BODY | {"items": [{"menu_item_id": str(ITEM), "quantity": 0}]}),
            ("post", f"/api/v1/restaurants/{RID}/orders",
             BODY | {"items": [{"menu_item_id": str(ITEM), "quantity": -3}]}),
            ("post", f"/api/v1/restaurants/{RID}/orders",
             BODY | {"items": [{"menu_item_id": str(ITEM), "quantity": 1.5}]}),
            ("post", f"/api/v1/restaurants/{RID}/orders",
             BODY | {"items": [{"menu_item_id": str(ITEM), "quantity": 1, "unit_price_paise": 1}]}),
            ("post", f"/api/v1/restaurants/{RID}/orders", BODY | {"total_paise": 1}),
            ("post", f"/api/v1/restaurants/{RID}/orders", BODY | {"customer_phone": None}),
            ("post", f"/api/v1/restaurants/{RID}/orders", {"items": BODY["items"]}),
            ("post", f"/api/v1/restaurants/{RID}/orders",
             BODY | {"items": [{"menu_item_id": "not-a-uuid", "quantity": 1}]}),
            ("get", f"/api/v1/restaurants/{RID}/orders?status=paid", None),
            ("get", f"/api/v1/restaurants/{RID}/orders?limit=101", None),
            ("patch", f"/api/v1/orders/{RID}/status", {"status": "paid"}),
            ("patch", f"/api/v1/orders/{RID}/status", {}),
            ("get", "/api/v1/orders/not-a-uuid", None),
        ],
    )  # fmt: skip
    def test_invalid_requests_rejected_before_database(self, client, db, method, path, body):
        response = client.request(method, path, json=body)
        assert response.status_code == 422
        db.add.assert_not_called()
        db.commit.assert_not_called()


class TestNotFound:
    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("post", f"/api/v1/restaurants/{RID}/orders", BODY),
            ("get", f"/api/v1/restaurants/{RID}/orders", None),
            ("get", f"/api/v1/orders/{RID}", None),
            ("patch", f"/api/v1/orders/{RID}/status", {"status": "confirmed"}),
        ],
    )
    def test_unknown_resources_return_404(self, client, db, method, path, body) -> None:
        db.get.return_value = None
        response = client.request(method, path, json=body)
        assert response.status_code == 404
        db.commit.assert_not_called()


# ── Service paths that need precise control over the database ────────────────


def _service_session(*, existing_after_conflict, commit_error) -> MagicMock:
    session = MagicMock(spec=Session)
    session.get.return_value = SimpleNamespace(is_active=True, currency="INR")
    session.scalar.side_effect = [None, existing_after_conflict]  # key lookups
    menu_item = SimpleNamespace(id=ITEM, name="Masala Dosa", price_paise=12000, is_available=True, is_active=True)
    # Menu items, then the (empty) stock rows locked for the reservation.
    session.scalars.return_value.all.side_effect = [[menu_item], []]
    session.commit.side_effect = commit_error
    return session


def _integrity_error(constraint: str) -> IntegrityError:
    orig = Exception(f"violates {constraint}; INSERT ... {SECRET} {PHONE}")
    orig.diag = SimpleNamespace(constraint_name=constraint)  # type: ignore[attr-defined]
    return IntegrityError("INSERT INTO orders ...", {"customer_phone": PHONE}, orig)


class TestConcurrentDuplicatePath:
    """Two requests with the same key both pass the lookup; the database's
    unique constraint rejects the second insert."""

    def test_same_request_returns_the_winning_order(self) -> None:
        winner = _fake_order()
        session = _service_session(
            existing_after_conflict=winner,
            commit_error=_integrity_error("uq_orders_restaurant_id_idempotency_key"),
        )
        result = order_service.create_order(session, RID, OrderCreate(**BODY))
        assert result.order is winner and result.created is False
        session.rollback.assert_called_once()

    def test_different_request_with_same_key_conflicts(self) -> None:
        session = _service_session(
            existing_after_conflict=_fake_order(request_hash="0" * 64),
            commit_error=_integrity_error("uq_orders_restaurant_id_idempotency_key"),
        )
        with pytest.raises(ConflictError, match="different order request"):
            order_service.create_order(session, RID, OrderCreate(**BODY))
        session.rollback.assert_called_once()


class TestRollbackOnFailure:
    def test_failed_commit_rolls_back_and_hides_details(self) -> None:
        session = _service_session(
            existing_after_conflict=None,
            commit_error=_integrity_error("ck_order_items_line_total_matches"),
        )
        with pytest.raises(ConflictError) as exc_info:
            order_service.create_order(session, RID, OrderCreate(**BODY))
        session.rollback.assert_called_once()
        assert exc_info.value.detail == "Order line totals are inconsistent."
        _assert_no_leak(str(exc_info.value.detail))

    def test_api_response_and_logs_do_not_leak(self, client, db, caplog) -> None:
        db.get.return_value = SimpleNamespace(is_active=True, currency="INR")
        db.scalar.side_effect = [None, None]
        db.scalars.return_value.all.side_effect = [
            [SimpleNamespace(id=ITEM, name="Masala Dosa", price_paise=12000, is_available=True, is_active=True)],
            [],  # no tracked stock to reserve
        ]
        db.commit.side_effect = _integrity_error("some_internal_constraint")
        with caplog.at_level("DEBUG"):
            response = client.post(f"/api/v1/restaurants/{RID}/orders", json=BODY)
        assert response.status_code == 409
        assert response.json() == {"detail": "The request conflicts with existing data."}
        _assert_no_leak(response.text)
        # Logging really happened (not vacuously empty), but without PII or secrets.
        assert "Integrity error on constraint some_internal_constraint" in caplog.text
        _assert_no_leak(caplog.text)
        assert "Asha" not in caplog.text
