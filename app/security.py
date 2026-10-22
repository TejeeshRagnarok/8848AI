"""Request authentication for staff endpoints and per-order customer tokens.

- Staff: header ``X-Staff-Key`` must equal ``STAFF_API_KEY``.  If no key is
  configured, staff endpoints answer 503 instead of running unauthenticated.
- Customers: placing an order returns a ``customer_token`` (an HMAC of the
  order id).  Sent back as ``X-Order-Token`` it grants access to that one
  order's tracking view and cancellation, and nothing else.
"""

import hashlib
import hmac
import logging
import secrets
import uuid
from typing import Annotated

from fastapi import Header, HTTPException, status

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
