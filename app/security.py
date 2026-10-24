"""Request authentication for staff endpoints and per-order customer tokens.

- Staff: header ``X-Staff-Key`` must equal ``STAFF_API_KEY``.  If no key is
  configured, staff endpoints answer 503 instead of running unauthenticated.
- Customers: placing an order returns a ``customer_token`` (an HMAC of the
  order id).  Sent back as ``X-Order-Token`` it grants access to that one
  order's tracking view and cancellation, and nothing else.
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Annotated

from fastapi import Header, HTTPException, Request, status

from app.config import settings

logger = logging.getLogger(__name__)

_process_secret = secrets.token_bytes(32)
_warned = False


def _token_secret() -> bytes:
    global _warned  # noqa: PLW0603
    if settings.order_token_secret is not None:
        return settings.order_token_secret.get_secret_value().encode()
    if not _warned:
        logger.warning("ORDER_TOKEN_SECRET is not set; order tokens stop working on restart")
        _warned = True
    return _process_secret


def order_token(order_id: uuid.UUID) -> str:
    return hmac.new(_token_secret(), order_id.bytes, hashlib.sha256).hexdigest()[:40]


def valid_order_token(order_id: uuid.UUID, token: str | None) -> bool:
    return bool(token) and hmac.compare_digest(order_token(order_id), token)


def is_staff(key: str | None) -> bool:
    expected = settings.staff_api_key
    if expected is None or not key:
        return False
    return hmac.compare_digest(key.encode(), expected.get_secret_value().encode())


# ── Dashboard sessions ───────────────────────────────────────────────────────
#
# Staff sign in to the dashboard once with the staff key; the browser then
# holds only a signed, HttpOnly cookie bound to ONE restaurant (scripts cannot
# read it, and the key is never stored client-side).  State-changing requests
# must also send ``X-Requested-With: dashboard`` (a cross-site page cannot set
# it without a CORS preflight, which this app does not allow).

DASHBOARD_COOKIE = "rva_dashboard"
DASHBOARD_COOKIE_PATH = "/api/v1/dashboard"
DASHBOARD_HEADER = "x-requested-with"
DASHBOARD_HEADER_VALUE = "dashboard"
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


@dataclass(frozen=True)
class DashboardSession:
    restaurant_id: uuid.UUID
    expires_at: int  # unix seconds


def _session_key() -> bytes:
    # Derived from the staff key: rotating STAFF_API_KEY signs everyone out.
    secret = settings.staff_api_key.get_secret_value() if settings.staff_api_key else ""
    return hashlib.sha256(b"dashboard-session:" + secret.encode()).digest()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue_dashboard_session(restaurant_id: uuid.UUID) -> tuple[str, int]:
    """A signed token (and its lifetime in seconds) for the session cookie."""
    lifetime = settings.dashboard_session_hours * 3600
    payload = _b64(json.dumps({"r": str(restaurant_id), "e": int(time.time()) + lifetime}).encode())
    signature = _b64(hmac.new(_session_key(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{signature}", lifetime


def read_dashboard_session(token: str | None) -> DashboardSession | None:
    if not token or settings.staff_api_key is None or token.count(".") != 1:
        return None
    payload, signature = token.split(".")
    expected = _b64(hmac.new(_session_key(), payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        data = json.loads(_unb64(payload))
        session = DashboardSession(restaurant_id=uuid.UUID(data["r"]), expires_at=int(data["e"]))
    except (ValueError, KeyError, TypeError):
        return None
    return session if session.expires_at > time.time() else None


def require_dashboard(request: Request) -> DashboardSession:
    """FastAPI dependency: a valid dashboard session (restaurant-scoped)."""
    if settings.staff_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The dashboard is disabled until STAFF_API_KEY is configured.",
        )
    session = read_dashboard_session(request.cookies.get(DASHBOARD_COOKIE))
    if session is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Please sign in.")
    if request.method not in _SAFE_METHODS and (
        request.headers.get(DASHBOARD_HEADER) != DASHBOARD_HEADER_VALUE
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Missing dashboard header.")
    return session


class LoginThrottle:
    """At most ``limit`` failed sign-ins per client per ``window`` seconds."""

    def __init__(self, limit: int = 5, window: float = 300.0) -> None:
        self._limit, self._window = limit, window
        self._failures: dict[str, list[float]] = {}

    def blocked(self, client: str) -> bool:
        now = time.monotonic()
        recent = [t for t in self._failures.get(client, []) if now - t < self._window]
        self._failures[client] = recent
        return len(recent) >= self._limit

    def failed(self, client: str) -> None:
        self._failures.setdefault(client, []).append(time.monotonic())

    def clear(self, client: str) -> None:
        self._failures.pop(client, None)


login_throttle = LoginThrottle()


def require_staff(x_staff_key: Annotated[str | None, Header()] = None) -> None:
    """FastAPI dependency for staff-only endpoints."""
    if settings.staff_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Staff endpoints are disabled until STAFF_API_KEY is configured.",
        )
    if not is_staff(x_staff_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="A valid X-Staff-Key header is required."
        )
