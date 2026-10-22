"""Schema building blocks shared across resources."""

from typing import Annotated, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

T = TypeVar("T")

# Whitespace-trimmed, non-empty strings.
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Phone = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=5, max_length=20, pattern=r"^\+?[0-9][0-9 -]*[0-9]$"
    ),
]

# Integer paise.  Must be positive and fit PostgreSQL INTEGER; strict so that
# floats (12.5) and numeric strings ("1200") are rejected rather than coerced.
PricePaise = Annotated[int, Field(strict=True, gt=0, le=2_147_483_647)]

# Hard cap on page size for list endpoints.
MAX_PAGE_SIZE = 100


class RequestSchema(BaseModel):
    """Base for request bodies: unknown fields (``id``, ``created_at``, …) are rejected."""

    model_config = ConfigDict(extra="forbid")


class PartialUpdateSchema(RequestSchema):
    """Base for PATCH bodies: only fields the client sends are applied.

    Fields listed in ``non_nullable`` may be omitted but not set to ``null``,
    because the matching database columns are NOT NULL.
    """

    non_nullable: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="after")
    def _reject_null_for_required_columns(self):
        nulls = sorted(
            f for f in self.model_fields_set & self.non_nullable if getattr(self, f) is None
        )
        if nulls:
            raise ValueError(f"fields cannot be null: {', '.join(nulls)}")
        return self

    def changes(self) -> dict:
        """Return only the fields explicitly set in the request."""
        return self.model_dump(exclude_unset=True)


class Page(BaseModel, Generic[T]):
    """A page of results from a list endpoint."""

    items: list[T]
    total: int
    limit: int
    offset: int
