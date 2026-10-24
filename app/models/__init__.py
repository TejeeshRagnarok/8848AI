"""ORM models.

Importing this package registers every model on ``Base.metadata`` (used by
Alembic) and lets string-based relationships between models resolve.
"""

from app.models.inventory import InventoryItem, InventoryMovement, MovementKind
from app.models.menu_item import MenuItem
from app.models.order import (
    CancelReason,
    InventoryState,
    Order,
    OrderItem,
    OrderSource,
    OrderStatus,
)
from app.models.payment import Payment, PaymentStatus
from app.models.restaurant import Restaurant

__all__ = [
    "CancelReason",
    "InventoryItem",
    "InventoryMovement",
    "InventoryState",
    "MenuItem",
    "MovementKind",
    "Order",
    "OrderItem",
    "OrderSource",
    "OrderStatus",
    "Payment",
    "PaymentStatus",
    "Restaurant",
]
