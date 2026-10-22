"""MenuItem ORM model."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, Text, Uuid, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.inventory import InventoryItem
    from app.models.restaurant import Restaurant


class MenuItem(TimestampMixin, Base):
    """A single orderable item on a restaurant's menu."""

    __tablename__ = "menu_items"
    __table_args__ = (CheckConstraint("price_paise > 0", name="price_paise_positive"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    restaurant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("restaurants.id", ondelete="RESTRICT"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(100))
    # Integer paise (1 INR = 100 paise) so totals are exact; never use floats.
    price_paise: Mapped[int] = mapped_column(Integer)
    is_available: Mapped[bool] = mapped_column(default=True, server_default=true())

    restaurant: Mapped[Restaurant] = relationship(back_populates="menu_items")
    # None: stock is not tracked (only ``is_available`` limits ordering).
    inventory: Mapped[InventoryItem | None] = relationship(
        lazy="selectin", passive_deletes=True, uselist=False
    )

    @property
    def stock_tracked(self) -> bool:
        return self.inventory is not None

    @property
    def available_quantity(self) -> int | None:
        return self.inventory.available if self.inventory is not None else None

    @property
    def orderable(self) -> bool:
        """Can be ordered now: switched on and, if tracked, at least one in stock."""
        return self.is_available and (self.inventory is None or self.inventory.available > 0)

    def __repr__(self) -> str:
        return f"<MenuItem {self.name!r} {self.price_paise}p>"
