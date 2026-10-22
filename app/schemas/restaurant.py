"""Request/response schemas for restaurants."""

import uuid
from datetime import datetime
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints

from app.schemas.common import Name, PartialUpdateSchema, Phone, RequestSchema


def _valid_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("must be an IANA timezone name, e.g. 'Asia/Kolkata'") from None
    return value


# These patterns mirror the database check constraints on ``restaurants``.
Slug = Annotated[
    str, StringConstraints(max_length=100, pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
]
Currency = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
Timezone = Annotated[str, StringConstraints(max_length=64), AfterValidator(_valid_timezone)]


class RestaurantCreate(RequestSchema):
    name: Name
    slug: Slug
    phone: Phone | None = None
    timezone: Timezone = "Asia/Kolkata"
    currency: Currency = "INR"


class RestaurantUpdate(PartialUpdateSchema):
    """Partial update.  ``is_active: true`` explicitly reactivates a restaurant."""

    non_nullable = frozenset({"name", "slug", "timezone", "currency", "is_active"})

    name: Name | None = None
    slug: Slug | None = None
    phone: Phone | None = None
    timezone: Timezone | None = None
    currency: Currency | None = None
    is_active: bool | None = None


class RestaurantRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str
    phone: str | None
    timezone: str
    currency: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
