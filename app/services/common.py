"""Domain errors and transaction helpers shared by the services."""

import logging
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """Base for errors whose ``detail`` is safe to return to API clients."""

    def __init__(self, detail: Any) -> None:
        super().__init__(detail)
        self.detail = detail


class NotFoundError(ServiceError):
    """The requested resource does not exist."""


class ConflictError(ServiceError):
    """The request conflicts with current state or a database constraint."""


class UnprocessableError(ServiceError):
    """The request is well-formed but refers to data that cannot be used.

    ``detail`` is a list of ``{"type", "loc", "msg"}`` entries, matching
    FastAPI's own 422 validation-error format.
    """


# Client-safe messages for known constraints (names come from Base's naming convention).
_CONSTRAINT_MESSAGES = {
    "uq_restaurants_slug": "A restaurant with this slug already exists.",
    "fk_menu_items_restaurant_id_restaurants": "The referenced restaurant does not exist.",
    "ck_menu_items_price_paise_positive": "price_paise must be a positive integer.",
    "ck_restaurants_slug_format": "slug has an invalid format.",
    "ck_restaurants_currency_format": "currency must be a three-letter uppercase code.",
    "uq_orders_restaurant_id_idempotency_key": "idempotency_key was already used for another order.",
    "ck_orders_status_valid": "Invalid order status.",
    "ck_order_items_quantity_positive": "quantity must be a positive integer.",
    "ck_order_items_line_total_matches": "Order line totals are inconsistent.",
}


def constraint_name(exc: IntegrityError) -> str | None:
    """Name of the violated constraint, as reported by PostgreSQL."""
    return getattr(getattr(exc.orig, "diag", None), "constraint_name", None)


def constraint_message(exc: IntegrityError) -> str:
    """Map an IntegrityError to a client-safe message, never the raw DB error."""
    name = constraint_name(exc)
    logger.warning("Integrity error on constraint %s", name)
    return _CONSTRAINT_MESSAGES.get(name, "The request conflicts with existing data.")


def commit(session: Session) -> None:
    """Commit; on a constraint violation roll back and raise ``ConflictError``.

    The database constraints stay the source of truth, so concurrent requests
    (e.g. two creates with the same slug) are still rejected safely.
    """
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise ConflictError(constraint_message(exc)) from exc
