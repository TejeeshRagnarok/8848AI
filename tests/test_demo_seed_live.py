"""The demo seed creates records through the services and is safe to re-run.
Runs inside the rolled-back test session, so nothing is left behind."""

import uuid

import pytest
from sqlalchemy import func, select

from app import demo_seed
from app.demo_seed import DEMO_MENU, seed
from app.models import InventoryItem, MenuItem
from app.schemas import RestaurantCreate

pytestmark = pytest.mark.live_db


def test_seed_is_idempotent_and_keeps_live_stock(session, monkeypatch) -> None:
    # A throwaway slug, so the real demo restaurant (if seeded) is never touched.
    slug = f"test-seed-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(demo_seed, "DEMO_RESTAURANT", RestaurantCreate(name="Seed Test", slug=slug))
    rid = seed(session)
    items = session.scalars(select(MenuItem).where(MenuItem.restaurant_id == rid)).all()
    assert {i.name for i in items} == {row[0] for row in DEMO_MENU}
    assert all(i.inventory is not None and i.inventory.on_hand > 0 for i in items)

    # Stock changes (e.g. sales) survive a re-run; nothing is duplicated.
    dosa = next(i for i in items if i.name == "Masala Dosa")
    dosa.inventory.on_hand = 7
    session.commit()
    assert seed(session) == rid
    count = session.scalar(select(func.count()).select_from(MenuItem).where(MenuItem.restaurant_id == rid))
    assert count == len(DEMO_MENU)
    assert session.get(InventoryItem, dosa.id).on_hand == 7

    seed(session, reset_stock=True)
    session.expire_all()
    assert session.get(InventoryItem, dosa.id).on_hand == 30
