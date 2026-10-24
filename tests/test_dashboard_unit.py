"""Dashboard session tokens (no database)."""

import time
import uuid

from pydantic import SecretStr

from app.config import settings
from app.security import issue_dashboard_session, read_dashboard_session


def test_token_round_trip_and_expiry(monkeypatch) -> None:
    monkeypatch.setattr(settings, "staff_api_key", SecretStr("k1"))
    rid = uuid.uuid4()
    token, lifetime = issue_dashboard_session(rid)
    session = read_dashboard_session(token)
    assert session.restaurant_id == rid and lifetime == settings.dashboard_session_hours * 3600
    monkeypatch.setattr(time, "time", lambda: session.expires_at + 1)
    assert read_dashboard_session(token) is None


def test_rotating_the_staff_key_signs_everyone_out(monkeypatch) -> None:
    monkeypatch.setattr(settings, "staff_api_key", SecretStr("k1"))
    token, _ = issue_dashboard_session(uuid.uuid4())
    monkeypatch.setattr(settings, "staff_api_key", SecretStr("k2"))
    assert read_dashboard_session(token) is None
    monkeypatch.setattr(settings, "staff_api_key", None)
    assert read_dashboard_session(token) is None


def test_garbage_tokens_are_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "staff_api_key", SecretStr("k1"))
    for token in (None, "", "abc", "a.b.c", "e30.sig", "not-base64!.x"):
        assert read_dashboard_session(token) is None
