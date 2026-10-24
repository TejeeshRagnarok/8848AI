"""Live concurrency tests for order creation against real PostgreSQL.

Concurrent transactions need committed data that other connections can see,
so unlike the other live tests these commit a throwaway restaurant and menu.
The ``committed_menu`` fixture always deletes everything it (and the orders
placed against it) created, by ID, and verifies nothing is left behind.
"""

import threading
import time
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import Engine, delete, func, select, text
from sqlalchemy.orm import Session

from app.models import (
    InventoryItem,
    InventoryMovement,
    MenuItem,
    Order,
    OrderItem,
    Payment,
    Restaurant,
)
from app.schemas import OrderCreate
from app.models import OrderSource
from app.models.payment import DemoOutcome
from app.schemas.order import WalkInOrderCreate
from app.services.common import ConflictError, UnprocessableError
from app.services.inventory import InsufficientStockError
from app.services.orders import create_order, expire_reservations
from app.services.payments import simulate

pytestmark = pytest.mark.live_db


def delete_restaurant_data(engine: Engine, rid: uuid.UUID) -> None:
    """Remove a committed test restaurant and everything that references it."""
    with engine.begin() as conn:
        order_ids = select(Order.id).where(Order.restaurant_id == rid)
        item_ids = select(MenuItem.id).where(MenuItem.restaurant_id == rid)
        conn.execute(delete(InventoryMovement).where(InventoryMovement.order_id.in_(order_ids)))
        conn.execute(delete(InventoryMovement).where(InventoryMovement.menu_item_id.in_(item_ids)))
        conn.execute(delete(Payment).where(Payment.order_id.in_(order_ids)))
        conn.execute(delete(OrderItem).where(OrderItem.order_id.in_(order_ids)))
        conn.execute(delete(Order).where(Order.restaurant_id == rid))
        conn.execute(delete(InventoryItem).where(InventoryItem.menu_item_id.in_(item_ids)))
        conn.execute(delete(MenuItem).where(MenuItem.restaurant_id == rid))
        conn.execute(delete(Restaurant).where(Restaurant.id == rid))


@pytest.fixture
def committed_menu(engine: Engine) -> Generator[dict, None, None]:
    rid, dosa, coffee, idli = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    with Session(engine) as s:
        s.add(Restaurant(id=rid, name="Concurrency Test", slug=f"test-conc-{uuid.uuid4().hex[:10]}"))
        s.flush()
        s.add_all([
            MenuItem(id=dosa, restaurant_id=rid, name="Masala Dosa", price_paise=12000),
            MenuItem(id=coffee, restaurant_id=rid, name="Filter Coffee", price_paise=4000),
            MenuItem(id=idli, restaurant_id=rid, name="Idli", price_paise=4000),
        ])  # fmt: skip
        s.flush()
        s.add(InventoryItem(menu_item_id=idli, on_hand=3, reserved=0, sold=0))  # scarce
        s.commit()
    try:
        yield {"restaurant": rid, "dosa": dosa, "coffee": coffee, "idli": idli}
    finally:
        delete_restaurant_data(engine, rid)
        with engine.connect() as conn:
            leftovers = conn.execute(
                text(
                    "SELECT (SELECT count(*) FROM restaurants WHERE id = :r)"
                    " + (SELECT count(*) FROM menu_items WHERE restaurant_id = :r)"
                    " + (SELECT count(*) FROM orders WHERE restaurant_id = :r)"
                ),
                {"r": rid},
            ).scalar_one()
        assert leftovers == 0, "concurrency test data was not cleaned up"


def _request(menu: dict, key: str, quantity: int = 1) -> OrderCreate:
    return OrderCreate(
        customer_phone="+91 98765 43210",
        items=[{"menu_item_id": str(menu["dosa"]), "quantity": quantity}],
        idempotency_key=key,
    )


def _place(engine: Engine, restaurant_id: uuid.UUID, data: OrderCreate, barrier=None):
    with Session(engine, expire_on_commit=False) as session:
        if barrier is not None:
            barrier.wait(timeout=10)
        try:
            result = create_order(session, restaurant_id, data)
            return result.order.id, result.created
        except Exception as exc:  # returned so the test thread can inspect it
            return exc


def _count_orders(engine: Engine, restaurant_id: uuid.UUID) -> int:
    with engine.connect() as conn:
        return conn.execute(
            select(func.count()).select_from(Order).where(Order.restaurant_id == restaurant_id)
        ).scalar_one()


class TestConcurrentSubmissions:
    def test_simultaneous_retries_with_same_key_create_one_order(self, engine, committed_menu) -> None:
        workers = 5
        barrier = threading.Barrier(workers)
        data = _request(committed_menu, "same-call")
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(
                lambda _: _place(engine, committed_menu["restaurant"], data, barrier), range(workers)
            ))  # fmt: skip

        assert all(isinstance(r, tuple) for r in results), results
        assert len({order_id for order_id, _ in results}) == 1
        assert sum(created for _, created in results) == 1
        assert _count_orders(engine, committed_menu["restaurant"]) == 1

    def test_simultaneous_distinct_orders_all_succeed(self, engine, committed_menu) -> None:
        workers = 5
        barrier = threading.Barrier(workers)
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(
                lambda n: _place(engine, committed_menu["restaurant"],
                                 _request(committed_menu, f"call-{n}"), barrier),
                range(workers),
            ))  # fmt: skip
        assert all(isinstance(r, tuple) and r[1] for r in results), results
        with engine.connect() as conn:
            numbers = conn.execute(
                select(Order.order_number).where(Order.restaurant_id == committed_menu["restaurant"])
            ).scalars().all()
        assert len(numbers) == len(set(numbers)) == workers


class TestMenuChangesDuringOrdering:
    """An uncommitted menu change holds the row lock; the order waits for it
    and then prices against the committed result, never a stale read."""

    def _race(self, engine, menu, update_sql: str) -> object:
        with engine.connect() as admin:
            admin.begin()
            admin.execute(text(update_sql), {"id": menu["dosa"]})  # row locked, not committed
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(_place, engine, menu["restaurant"], _request(menu, "race", 2))
                time.sleep(0.5)
                assert not future.done(), "order creation should wait for the menu row lock"
                admin.commit()
                return future.result(timeout=10)

    def test_item_made_unavailable_mid_order_is_rejected(self, engine, committed_menu) -> None:
        result = self._race(
            engine, committed_menu, "UPDATE menu_items SET is_available = false WHERE id = :id"
        )
        assert isinstance(result, UnprocessableError)
        assert result.detail[0]["type"] == "menu_item_unavailable"
        assert _count_orders(engine, committed_menu["restaurant"]) == 0

    def test_price_changed_mid_order_uses_committed_price(self, engine, committed_menu) -> None:
        result = self._race(
            engine, committed_menu, "UPDATE menu_items SET price_paise = 15000 WHERE id = :id"
        )
        order_id, created = result
        assert created
        with Session(engine) as s:
            order = s.get(Order, order_id)
            assert (order.items[0].unit_price_paise, order.total_paise) == (15000, 30000)


# ── Inventory under concurrency ──────────────────────────────────────────────


def _idli_order(menu: dict, key: str, quantity: int = 1) -> OrderCreate:
    return OrderCreate(
        customer_phone="+91 98765 43210",
        items=[{"menu_item_id": str(menu["idli"]), "quantity": quantity}],
        idempotency_key=key,
    )


def _stock(engine: Engine, item_id: uuid.UUID) -> tuple[int, int, int]:
    with engine.connect() as conn:
        return tuple(conn.execute(
            select(InventoryItem.on_hand, InventoryItem.reserved, InventoryItem.sold)
            .where(InventoryItem.menu_item_id == item_id)
        ).one())  # fmt: skip


def _movement_count(engine: Engine, order_id: uuid.UUID, kind: str) -> int:
    with engine.connect() as conn:
        return conn.execute(
            select(func.count()).select_from(InventoryMovement)
            .where(InventoryMovement.order_id == order_id, InventoryMovement.kind == kind)
        ).scalar_one()  # fmt: skip


def _in_session(engine: Engine, fn, barrier=None):
    with Session(engine, expire_on_commit=False) as session:
        if barrier is not None:
            barrier.wait(timeout=10)
        try:
            return fn(session)
        except Exception as exc:  # returned so the test thread can inspect it
            return exc


class TestInventoryRaces:
    def test_customers_competing_for_limited_stock_never_oversell(self, engine, committed_menu) -> None:
        workers = 8  # 8 customers, 3 idlis
        barrier = threading.Barrier(workers)
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(
                lambda n: _place(engine, committed_menu["restaurant"],
                                 _idli_order(committed_menu, f"idli-{n}"), barrier),
                range(workers),
            ))  # fmt: skip
        placed = [r for r in results if isinstance(r, tuple)]
        refused = [r for r in results if isinstance(r, InsufficientStockError)]
        assert len(placed) == 3 and len(refused) == 5, results
        assert _stock(engine, committed_menu["idli"]) == (3, 3, 0)
        assert _count_orders(engine, committed_menu["restaurant"]) == 3

    def test_duplicate_payment_callbacks_sell_once(self, engine, committed_menu) -> None:
        order_id, _ = _place(engine, committed_menu["restaurant"], _idli_order(committed_menu, "pay", 2))
        with Session(engine) as s:
            payment = s.scalar(select(Payment).where(Payment.order_id == order_id))
            payment_id, amount = payment.id, payment.amount_paise
        workers = 5
        barrier = threading.Barrier(workers)
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(
                lambda _: _in_session(engine, lambda s: simulate(s, payment_id, DemoOutcome.SUCCEEDED, amount),
                                      barrier),
                range(workers),
            ))  # fmt: skip
        assert all(not isinstance(r, Exception) for r in results), results
        assert sum(not r.replayed for r in results) == 1
        assert _stock(engine, committed_menu["idli"]) == (1, 0, 2)
        assert _movement_count(engine, order_id, "sell") == 1

    def test_payment_racing_expiry_has_exactly_one_outcome(self, engine, committed_menu) -> None:
        order_id, _ = _place(engine, committed_menu["restaurant"], _idli_order(committed_menu, "race", 1))
        with engine.begin() as conn:
            conn.execute(text("UPDATE orders SET reservation_expires_at = now() - interval '1 second' "
                              "WHERE id = :id"), {"id": order_id})  # fmt: skip
        with Session(engine) as s:
            payment_id = s.scalar(select(Payment.id).where(Payment.order_id == order_id))
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(2) as pool:
            pay = pool.submit(_in_session, engine,
                              lambda s: simulate(s, payment_id, DemoOutcome.SUCCEEDED, 4000), barrier)
            sweep = pool.submit(_in_session, engine, lambda s: expire_reservations(s), barrier)
            pay_result, sweep_result = pay.result(timeout=15), sweep.result(timeout=15)
        # Expired before payment: payment is refused whichever ran first.
        assert isinstance(pay_result, ConflictError), pay_result
        assert not isinstance(sweep_result, Exception), sweep_result
        assert _stock(engine, committed_menu["idli"]) == (3, 0, 0)
        assert _movement_count(engine, order_id, "release") == 1
        assert _movement_count(engine, order_id, "sell") == 0
        with Session(engine) as s:
            order = s.get(Order, order_id)
            assert (order.status, order.cancel_reason, order.payment.status) == ("cancelled", "expired", "expired")

    def test_voice_and_counter_racing_for_the_last_item(self, engine, committed_menu) -> None:
        with engine.begin() as conn:
            conn.execute(text("UPDATE inventory_items SET on_hand = 1 WHERE menu_item_id = :id"),
                         {"id": committed_menu["idli"]})  # fmt: skip
        barrier = threading.Barrier(2)

        def place(source):
            return _in_session(engine, lambda s: create_order(
                s, committed_menu["restaurant"],
                WalkInOrderCreate(items=[{"menu_item_id": str(committed_menu["idli"]), "quantity": 1}],
                                  idempotency_key=f"last-{source.value}"),
                source=source), barrier)  # fmt: skip

        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(place, [OrderSource.AI_VOICE, OrderSource.WALK_IN_POS]))
        winners = [r for r in results if not isinstance(r, Exception)]
        losers = [r for r in results if isinstance(r, InsufficientStockError)]
        assert len(winners) == 1 and len(losers) == 1, results
        assert _stock(engine, committed_menu["idli"]) == (1, 1, 0)
        with Session(engine) as s:
            sources = s.scalars(select(Order.source).where(Order.restaurant_id == committed_menu["restaurant"])).all()
        assert len(sources) == 1 and sources[0] in ("AI_VOICE", "WALK_IN_POS")
