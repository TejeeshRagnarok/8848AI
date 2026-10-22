"""Tests for the /health endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    """GET /health must respond with HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_contains_status_ok() -> None:
    """GET /health body must include {"status": "ok"}."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"


def test_health_response_contains_app_name() -> None:
    """GET /health body must include the configured application name."""
    response = client.get("/health")
    data = response.json()
    assert "app" in data
    assert len(data["app"]) > 0


def test_health_response_contains_env() -> None:
    """GET /health body must include the current environment name."""
    response = client.get("/health")
    data = response.json()
    assert "env" in data
