"""SQLAlchemy engine, session factory, declarative base, and FastAPI dependency.

This module provides:
- ``engine``          – The SQLAlchemy ``Engine`` (sync, Psycopg 3 driver).
- ``SessionLocal``    – A configured ``sessionmaker`` bound to the engine.
- ``Base``            – Declarative base class for ORM models.
- ``get_db()``        – FastAPI dependency that yields a session per request.

The engine is **lazily created** on first use so the application can start
and serve ``/health`` even when ``DATABASE_URL`` is not configured.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING

from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)

# ── Declarative base & exceptions ───────────────────────────────────────────


class DatabaseUnavailableError(RuntimeError):
    """Raised when the database is unreachable or DATABASE_URL is not configured."""


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models.

    Import and subclass this in every ``app/models/*.py`` module so Alembic's
    ``target_metadata`` picks up all tables automatically.

    The naming convention gives every index and constraint a deterministic
    name, so migrations can reliably refer to (and drop) them later.
    """

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


# ── Engine (lazy singleton) ──────────────────────────────────────────────────

_engine: Engine | None = None


def get_engine() -> Engine:
    """Return the SQLAlchemy engine, creating it on first call.

    Raises ``DatabaseUnavailableError`` if ``DATABASE_URL`` is not configured.
    """
    global _engine  # noqa: PLW0603
    if _engine is not None:
        return _engine

    if not settings.database_url:
        raise DatabaseUnavailableError(
            "DATABASE_URL is not set. "
            "Configure it in .env or as an environment variable."
        )

    _engine = create_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_pool_max_overflow,
        pool_pre_ping=True,
        echo=settings.db_echo_sql,
        connect_args={"connect_timeout": settings.db_connect_timeout},
    )
    # Log only host/db, never credentials.
    url_obj = _engine.url
    logger.info(
        "Database engine created → %s@%s:%s/%s",
        url_obj.username,
        url_obj.host,
        url_obj.port,
        url_obj.database,
    )
    return _engine


# ── Session factory ──────────────────────────────────────────────────────────


def get_session_factory() -> sessionmaker[Session]:
    """Return a ``sessionmaker`` bound to the current engine."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


# ── FastAPI dependency ───────────────────────────────────────────────────────


def get_db() -> Generator[Session, None, None]:
    """Yield a database session, ensuring it is closed after the request."""
    try:
        factory = get_session_factory()
        session = factory()
    except Exception as exc:
        if isinstance(exc, DatabaseUnavailableError):
            raise
        raise DatabaseUnavailableError(
            "Database session could not be established."
        ) from exc

    try:
        yield session
    finally:
        session.close()
