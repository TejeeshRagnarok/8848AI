"""Shared fixtures for live PostgreSQL tests (``live_db`` marker).

The ``engine`` fixture skips when the database is unreachable or not migrated
to head.  Every test runs inside an outer transaction that is always rolled
back, so nothing is ever written to the real database.
"""

from collections.abc import Generator
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from fastapi.testclient import TestClient
from sqlalchemy import Connection, Engine, create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.main import app
from app.security import require_staff

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def staff_access() -> Generator[None, None, None]:
    """API tests act as staff by default; authentication tests remove this
    override (``app.dependency_overrides.pop(require_staff)``)."""
    app.dependency_overrides[require_staff] = lambda: None
    try:
        yield
    finally:
        app.dependency_overrides.pop(require_staff, None)


@pytest.fixture(scope="session")
def engine() -> Generator[Engine, None, None]:
    if not settings.database_url:
        pytest.skip("DATABASE_URL is not configured")
    # A dedicated engine (same app settings) keeps app.database's cached
    # engine untouched for the other tests.
    eng = create_engine(
        settings.database_url,
        connect_args={"connect_timeout": settings.db_connect_timeout},
    )
    try:
        with eng.connect() as conn:
            current = MigrationContext.configure(conn).get_current_revision()
    except Exception:
        eng.dispose()
        pytest.skip("PostgreSQL is not reachable")
    head = ScriptDirectory.from_config(Config(str(PROJECT_ROOT / "alembic.ini"))).get_current_head()
    if current != head:
        eng.dispose()
        pytest.skip(f"database is at {current}, not head {head}; run alembic upgrade head")
    yield eng
    eng.dispose()


@pytest.fixture
def connection(engine: Engine) -> Generator[Connection, None, None]:
    conn = engine.connect()
    outer = conn.begin()
    try:
        yield conn
    finally:
        outer.rollback()  # discard everything the test did
        conn.close()


@pytest.fixture
def session(connection: Connection) -> Generator[Session, None, None]:
    # Session commits/rollbacks become SAVEPOINT operations inside the outer
    # transaction, so a failed flush can be recovered within one test.
    with Session(bind=connection, join_transaction_mode="create_savepoint") as s:
        yield s


@pytest.fixture
def live_client(session: Session) -> Generator[TestClient, None, None]:
    """A TestClient whose requests use the rolled-back test session."""

    def override_get_db() -> Generator[Session, None, None]:
        yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)
