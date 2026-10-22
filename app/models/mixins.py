"""Reusable column mixins for ORM models."""

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column


class TimestampMixin:
    """Adds timezone-aware ``created_at`` / ``updated_at`` columns.

    Both are set by PostgreSQL.  ``statement_timestamp()`` (rather than
    ``now()``, which is fixed for the whole transaction) means an update made
    later in the same transaction as the insert still advances ``updated_at``.
    ``updated_at`` is refreshed on every ORM update; raw SQL ``UPDATE``
    statements must set it themselves.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.statement_timestamp(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.statement_timestamp(),
        onupdate=func.statement_timestamp(),
    )
