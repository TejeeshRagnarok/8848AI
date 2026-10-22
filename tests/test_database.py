"""Tests for database configuration, session dependency, and /health/db endpoint.

All tests run without a live PostgreSQL server — they use mocks and
FastAPI dependency overrides.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings
from app.database import Base, get_db
from app.main import app

client = TestClient(app)


# ── Configuration tests ──────────────────────────────────────────────────────


class TestDatabaseConfig:
    """Verify database URL handling in Settings."""

    def test_database_url_defaults_to_none(self) -> None:
        """When DATABASE_URL is absent, settings.database_url is None."""
        s = Settings(
            _env_file=None,  # type: ignore[call-arg]
            database_url=None,
        )
        assert s.database_url is None

    def test_database_url_reads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DATABASE_URL should be picked up from the environment."""
        url = "postgresql+psycopg://user:pass@localhost:5432/testdb"
        monkeypatch.setenv("DATABASE_URL", url)
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.database_url == url

    def test_database_url_with_special_chars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Percent-encoded special characters in the password must survive."""
        url = "postgresql+psycopg://user:p%40ss%23word@localhost:5432/testdb"
        monkeypatch.setenv("DATABASE_URL", url)
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.database_url == url

    def test_pool_defaults(self) -> None:
        """Pool-tuning fields have sensible defaults."""
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.db_pool_size == 5
        assert s.db_pool_max_overflow == 10
        assert s.db_echo_sql is False
        assert s.db_connect_timeout == 5


# ── Declarative Base tests ───────────────────────────────────────────────────


class TestDeclarativeBase:
    """Verify the shared Base is usable."""

    def test_base_has_metadata(self) -> None:
        """Base.metadata must exist for Alembic autogenerate."""
        assert Base.metadata is not None


# ── Session dependency tests ─────────────────────────────────────────────────


class TestGetDbDependency:
    """Verify get_db yields and closes a session."""

    def test_get_db_yields_and_closes(self) -> None:
        """get_db must yield a Session and call close() in the finally block."""
        mock_session = MagicMock(spec=Session)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("app.database.get_session_factory", return_value=mock_factory):
            gen = get_db()
            session = next(gen)
            assert session is mock_session

            # Simulate request completion — the finally block runs
            try:
                next(gen)
            except StopIteration:
                pass

            mock_session.close.assert_called_once()

    def test_get_db_closes_on_exception(self) -> None:
        """Session must be closed even if the request handler raises."""
        mock_session = MagicMock(spec=Session)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("app.database.get_session_factory", return_value=mock_factory):
            gen = get_db()
            next(gen)

            # Simulate an exception during request handling
            with pytest.raises(ValueError, match="boom"):
                gen.throw(ValueError("boom"))

            mock_session.close.assert_called_once()


# ── /health/db endpoint tests ────────────────────────────────────────────────


def _mock_db_ok() -> Session:
    """Return a mock session where execute(SELECT 1) succeeds."""
    session = MagicMock(spec=Session)
    session.execute.return_value = None
    return session


def _mock_db_fail() -> Session:
    """Return a mock session where execute raises an OperationalError."""
    session = MagicMock(spec=Session)
    session.execute.side_effect = ConnectionError("connection refused")
    return session


class TestHealthDbEndpoint:
    """Test /health/db with dependency overrides — no live database."""

    def test_health_db_returns_200_when_connected(self) -> None:
        """Successful SELECT 1 → HTTP 200."""
        mock_session = _mock_db_ok()

        def override_get_db():
            yield mock_session

        app.dependency_overrides[get_db] = override_get_db
        try:
            response = client.get("/health/db")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["database"] == "connected"
        finally:
            app.dependency_overrides.clear()

    def test_health_db_returns_503_when_unreachable(self) -> None:
        """Database error → HTTP 503 without leaking credentials."""
        mock_session = _mock_db_fail()

        def override_get_db():
            yield mock_session

        app.dependency_overrides[get_db] = override_get_db
        try:
            response = client.get("/health/db")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "error"
            assert data["database"] == "unreachable"
            # Must not leak credentials or stack traces in the response body
            body_text = response.text
            assert "password" not in body_text.lower()
            assert "connection refused" not in body_text
        finally:
            app.dependency_overrides.clear()

    def test_health_db_response_does_not_leak_url(self) -> None:
        """Error responses must not contain the database URL."""
        mock_session = MagicMock(spec=Session)
        mock_session.execute.side_effect = Exception(
            "could not connect to postgresql+psycopg://user:secret@host/db"
        )

        def override_get_db():
            yield mock_session

        app.dependency_overrides[get_db] = override_get_db
        try:
            response = client.get("/health/db")
            assert response.status_code == 503
            body_text = response.text
            assert "secret" not in body_text
            assert "psycopg" not in body_text
        finally:
            app.dependency_overrides.clear()

    def test_health_db_returns_503_when_no_database_configured(self) -> None:
        """When DATABASE_URL is not set and no override exists, /health/db returns 503."""
        app.dependency_overrides.clear()
        with patch("app.database.settings.database_url", None):
            response = client.get("/health/db")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "error"
            assert data["database"] == "unreachable"
