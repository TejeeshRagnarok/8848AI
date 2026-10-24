"""Request/response schemas for menu items."""

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

from app.schemas.common import Name, PartialUpdateSchema, PricePaise, RequestSchema, ShortText

Description = Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)]


class MenuItemCreate(RequestSchema):
    # restaurant_id comes from the URL path, never from the body.
    name: Name
    description: Description | None = None
    category: ShortText | None = None
    price_paise: PricePaise
    is_available: bool = True


class MenuItemUpdate(PartialUpdateSchema):
    """Partial update.  Items cannot be moved to another restaurant."""

    non_nullable = frozenset({"name", "price_paise", "is_available"})

    name: Name | None = None
    description: Description | None = None
    category: ShortText | None = None
    price_paise: PricePaise | None = None
    is_available: bool | None = None


class MenuItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    restaurant_id: uuid.UUID
    name: str
    description: str | None
    category: str | None
    price_paise: int
    is_available: bool
    is_active: bool = True  # False: retired from the menu (history kept)
    # Stock (computed from inventory): untracked items have no quantity.
    stock_tracked: bool = False
    available_quantity: int | None = None
    orderable: bool | None = None  # switched on and, if tracked, in stock
    created_at: datetime
    updated_at: datetime
