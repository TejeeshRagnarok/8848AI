"""Restaurant ORM model."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, String, Uuid, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.menu_item import MenuItem


class Restaurant(TimestampMixin, Base):
    """A restaurant whose calls and menu the agent handles."""

    __tablename__ = "restaurants"
    __table_args__ = (
        # Lowercase words separated by single hyphens, e.g. "spice-garden-2".
        CheckConstraint(r"slug ~ '^[a-z0-9]+(-[a-z0-9]+)*$'", name="slug_format"),
        # ISO 4217 alphabetic code, e.g. "INR".
        CheckConstraint("currency ~ '^[A-Z]{3}$'", name="currency_format"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(100), unique=True)
    phone: Mapped[str | None] = mapped_column(String(20))
    timezone: Mapped[str] = mapped_column(
        String(64), default="Asia/Kolkata", server_default="Asia/Kolkata"
    )
    currency: Mapped[str] = mapped_column(String(3), default="INR", server_default="INR")
    is_active: Mapped[bool] = mapped_column(default=True, server_default=true())

    # passive_deletes="all": the ORM never nulls out or deletes menu items when
    # a restaurant is deleted; the database's ON DELETE RESTRICT rejects it.
    menu_items: Mapped[list[MenuItem]] = relationship(
        back_populates="restaurant", passive_deletes="all"
    )

    def __repr__(self) -> str:
        return f"<Restaurant {self.slug!r}>"
