"""Create (or reuse) the Spice Garden Demo restaurant, its menu and stock.

    python -m app.demo_seed                # create missing records; keep existing stock
    python -m app.demo_seed --reset-stock  # also set every item's on-hand count to the demo level

Everything goes through the application services (same validation as the
API).  Re-running never duplicates records: the restaurant is found by slug
and items by name.  Stock is only set for items not yet tracked unless
``--reset-stock`` is given, and resetting never drops below units reserved
for unpaid orders.  Prints the restaurant id for ``EXOTEL_RESTAURANT_ID``.
"""

import argparse
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_session_factory
from app.models import MenuItem, Restaurant
from app.schemas import MenuItemCreate, RestaurantCreate
from app.services import inventory as inventory_service
from app.services import menu_items as menu_service
from app.services import restaurants as restaurant_service

DEMO_RESTAURANT = RestaurantCreate(
    name="Spice Garden Demo", slug="spice-garden-demo", timezone="Asia/Kolkata", currency="INR"
)

# (name, category, price in paise, description, demo stock)
DEMO_MENU: list[tuple[str, str, int, str, int]] = [
    ("Idli", "Breakfast", 4000, "Two steamed rice cakes with sambar and coconut chutney.", 60),
    ("Plain Dosa", "Dosa", 6000, "Crisp rice and lentil crepe with sambar and chutney.", 40),
    ("Masala Dosa", "Dosa", 8000, "Crisp dosa filled with spiced potato masala.", 30),
    ("Medu Vada", "Breakfast", 5000, "Two crisp lentil doughnuts with sambar and chutney.", 40),
    ("Upma", "Breakfast", 5000, "Semolina cooked with vegetables, mustard and curry leaves.", 20),
    ("Veg Meals", "Meals", 15000, "Rice, sambar, rasam, two vegetables, curd, papad and pickle.", 15),
    ("Tea", "Beverages", 2000, "Hot masala chai.", 100),
    ("Filter Coffee", "Beverages", 2500, "South Indian filter coffee with milk.", 100),
]  # fmt: skip


def seed(session: Session, *, reset_stock: bool = False) -> uuid.UUID:
    restaurant = session.scalar(select(Restaurant).where(Restaurant.slug == DEMO_RESTAURANT.slug))
    if restaurant is None:
        restaurant = restaurant_service.create_restaurant(session, DEMO_RESTAURANT)
        print(f"created restaurant {restaurant.name}")
    else:
        print(f"reusing restaurant {restaurant.name}")

    existing = {
        item.name: item
        for item in session.scalars(select(MenuItem).where(MenuItem.restaurant_id == restaurant.id))
    }
    for name, category, price, description, stock in DEMO_MENU:
        item = existing.get(name)
        if item is None:
            item = menu_service.create_menu_item(
                session,
                restaurant.id,
                MenuItemCreate(name=name, category=category, price_paise=price, description=description),
            )
            print(f"  created {name}")
        if item.inventory is None or reset_stock:
            target = max(stock, item.inventory.reserved) if item.inventory else stock
            inventory_service.set_stock(session, item.id, on_hand=target, note="demo seed")
            print(f"  stock {name}: {target} on hand")
    return restaurant.id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset-stock", action="store_true", help="reset on-hand counts to demo levels")
    args = parser.parse_args()
    with get_session_factory()() as session:
        restaurant_id = seed(session, reset_stock=args.reset_stock)
    print(f"\nEXOTEL_RESTAURANT_ID={restaurant_id}")


if __name__ == "__main__":
    main()
