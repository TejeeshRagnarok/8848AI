"""The agent's tools: their JSON schemas (shown to the model), strict argument
models, and an executor that runs them through the existing backend services.

Safety properties:
- Only the four tools below exist; any other name is refused.
- The restaurant is bound by the server (from the URL); it is not a tool
  argument, so the model cannot reach another restaurant's data.
- Arguments are parsed with ``extra="forbid"`` models: unknown fields such as
  prices, totals, ``restaurant_id`` or a "confirmed" flag are rejected.
- ``create_order`` never places an order.  It prices and stock-checks the
  request through the order engine (live PostgreSQL data, nothing reserved)
  and stores a server-side proposal; the order is placed, and its stock
  reserved, only when the customer explicitly confirms (see ``service.py``).
- ``get_order_status`` reads only orders placed in this conversation, and is
  the only source of truth for order and payment status.  What it verifies is
  recorded in ``verified_claims`` so the reply guard can allow those claims.
- Results are plain JSON data returned to the model as tool messages.
"""

import json
import logging
import uuid
from typing import Annotated, Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from app.agent.sessions import AgentSession, PendingOrder
from app.models import Restaurant
from app.schemas import OrderCreate
from app.schemas.common import Phone, RequestSchema, ShortText
from app.models.payment import DEMO_NOTICE
from app.schemas.order import MAX_LINES_PER_ORDER, Notes, OrderLineRequest
from app.services import menu_items as menu_service
from app.services import orders as order_service
from app.services.common import ConflictError, NotFoundError, UnprocessableError

logger = logging.getLogger(__name__)

MAX_MENU_RESULTS = 25
MAX_DESCRIPTION_CHARS = 300
MAX_ALTERNATIVES = 5


def format_money(paise: int, currency: str) -> str:
    """Exact integer formatting, e.g. 24150 -> 'INR 241.50' (no floats)."""
    sign = "-" if paise < 0 else ""
    rupees, cents = divmod(abs(paise), 100)
    return f"{currency} {sign}{rupees:,}.{cents:02d}"


# ── Argument models ──────────────────────────────────────────────────────────


class GetMenuArgs(RequestSchema):
    search: ShortText | None = None
    category: ShortText | None = None
    available_only: bool = True
    limit: Annotated[int, Field(ge=1, le=MAX_MENU_RESULTS)] = 10


class GetMenuItemArgs(RequestSchema):
    menu_item_id: uuid.UUID


class CreateOrderArgs(RequestSchema):
    customer_name: ShortText | None = None
    customer_phone: Phone
    items: Annotated[list[OrderLineRequest], Field(min_length=1, max_length=MAX_LINES_PER_ORDER)]
    notes: Notes | None = None


class GetOrderStatusArgs(RequestSchema):
    order_number: Annotated[str, Field(pattern=r"^ORD-[0-9]{1,12}$")] | None = None


ARG_MODELS: dict[str, type[BaseModel]] = {
    "get_menu": GetMenuArgs,
    "get_menu_item": GetMenuItemArgs,
    "create_order": CreateOrderArgs,
    "get_order_status": GetOrderStatusArgs,
}

# Hand-written (rather than generated) so the schemas stay plain JSON Schema
# that any OpenAI-compatible provider accepts.  A unit test keeps them in sync
# with the argument models above.
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_menu",
            "description": (
                "Search this restaurant's menu. Returns items with IDs, prices, availability and "
                "live stock (available_quantity; null means not stock-limited)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search": {"type": "string", "maxLength": 100,
                               "description": "Words to match in the item name or description."},
                    "category": {"type": "string", "maxLength": 100,
                                 "description": "Exact category name, e.g. 'Breakfast'."},
                    "available_only": {"type": "boolean", "default": True,
                                       "description": "Only items that can be ordered right now."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_MENU_RESULTS, "default": 10},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_menu_item",
            "description": "Get one menu item of this restaurant by its ID.",
            "parameters": {
                "type": "object",
                "properties": {"menu_item_id": {"type": "string", "description": "Item ID from get_menu."}},
                "required": ["menu_item_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": (
                "Prepare an order for the customer to review. This does NOT place the order or "
                "reserve stock: it checks availability and stock and returns the authoritative "
                "items and total, which you must read back to the customer. The order is placed "
                "only after the customer confirms. Call again with the full updated order if "
                "anything changes (items added, removed or changed)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string", "maxLength": 100},
                    "customer_phone": {"type": "string", "maxLength": 20,
                                       "description": "Customer's phone number, e.g. '+91 98765 43210'."},
                    "items": {
                        "type": "array", "minItems": 1, "maxItems": MAX_LINES_PER_ORDER,
                        "items": {
                            "type": "object",
                            "properties": {
                                "menu_item_id": {"type": "string"},
                                "quantity": {"type": "integer", "minimum": 1, "maximum": 100},
                            },
                            "required": ["menu_item_id", "quantity"],
                            "additionalProperties": False,
                        },
                    },
                    "notes": {"type": "string", "maxLength": 500},
                },
                "required": ["customer_phone", "items"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Look up the current status and (demo) payment status of orders placed in this "
                "conversation. Use it whenever the customer asks about their order or payment; "
                "never guess either."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {"type": "string", "pattern": "^ORD-[0-9]{1,12}$",
                                     "description": "Optional, e.g. 'ORD-100001'. Omit for all."},
                },
                "additionalProperties": False,
            },
        },
    },
]  # fmt: skip


# ── Result shaping ───────────────────────────────────────────────────────────


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"error": code, "message": message, **extra}


def _menu_item(item, currency: str) -> dict[str, Any]:
    description = item.description
    if description and len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS] + "…"
    return {
        "menu_item_id": str(item.id),
        "name": item.name,
        "description": description,
        "category": item.category,
        "price_paise": item.price_paise,
        "price": format_money(item.price_paise, currency),
        "is_available": item.is_available,
        "available_quantity": item.available_quantity,  # None: not stock-limited
        "orderable": item.orderable,
    }


def proposal_result(pending: PendingOrder, currency: str) -> dict[str, Any]:
    """The proposal as shown to both the model and the API client."""
    return {
        "proposal_id": pending.proposal_id,
        "items": [
            {
                "menu_item_id": str(line.menu_item_id),
                "name": line.item_name,
                "quantity": line.quantity,
                "unit_price_paise": line.unit_price_paise,
                "line_total_paise": line.line_total_paise,
            }
            for line in pending.quote.lines
        ],
        "subtotal_paise": pending.quote.subtotal_paise,
        "total_paise": pending.quote.total_paise,
        "total": format_money(pending.quote.total_paise, currency),
        "currency": currency,
        "customer_name": pending.customer_name,
        "customer_phone": pending.customer_phone,
        "notes": pending.notes,
    }


def _validation_details(exc: ValidationError) -> list[dict[str, Any]]:
    # Location and message only; never echo the offending input back.
    return [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]


# ── Executor ─────────────────────────────────────────────────────────────────


class ToolExecutor:
    """Runs tool calls for one restaurant within one request's DB session."""

    def __init__(self, db: Session, restaurant: Restaurant, session: AgentSession) -> None:
        self._db = db
        self._restaurant = restaurant
        self._session = session
        # Claims the reply guard may allow because a tool read them from the DB.
        self.verified_claims: set[str] = set()
        # Order statuses read by get_order_status this turn (for the guard).
        self.order_statuses: list[dict[str, Any]] = []
        self._handlers = {
            "get_menu": self._get_menu,
            "get_menu_item": self._get_menu_item,
            "create_order": self._create_order,
            "get_order_status": self._get_order_status,
        }

    def execute(self, name: str, raw_arguments: str) -> dict[str, Any]:
        """Validate and run one tool call.  Never raises for bad model input."""
        handler = self._handlers.get(name)
        if handler is None:
            logger.warning("Agent requested unknown tool")
            return _error("unknown_tool", "That tool does not exist. Use only the provided tools.")
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError):
            return _error("invalid_arguments", "Arguments must be a JSON object.")
        if not isinstance(arguments, dict):
            return _error("invalid_arguments", "Arguments must be a JSON object.")
        try:
            args = ARG_MODELS[name].model_validate(arguments)
        except ValidationError as exc:
            return _error("invalid_arguments", "Arguments failed validation.", details=_validation_details(exc))
        result = handler(args)
        logger.info("Agent tool %s -> %s", name, result.get("error") or result.get("status", "ok"))
        return result

    # Each handler calls the existing services, scoped to this restaurant.

    def _get_menu(self, args: GetMenuArgs) -> dict[str, Any]:
        items, total = menu_service.list_menu_items(
            self._db,
            self._restaurant.id,
            limit=args.limit,
            offset=0,
            is_available=True if args.available_only else None,
            in_stock=True if args.available_only else None,
            search=args.search,
            category=args.category,
        )
        currency = self._restaurant.currency
        return {
            "status": "ok",
            "items": [_menu_item(item, currency) for item in items],
            "returned": len(items),
            "total_matching": total,
        }

    def _get_menu_item(self, args: GetMenuItemArgs) -> dict[str, Any]:
        try:
            item = menu_service.get_menu_item(self._db, args.menu_item_id)
        except NotFoundError:
            item = None
        if item is None or item.restaurant_id != self._restaurant.id:
            return _error("not_found", "No menu item with that ID on this restaurant's menu.")
        return {"status": "ok", "item": _menu_item(item, self._restaurant.currency)}

    def _create_order(self, args: CreateOrderArgs) -> dict[str, Any]:
        try:
            # Same validation the order API applies (e.g. no duplicate items).
            request = OrderCreate(**args.model_dump())
        except ValidationError as exc:
            return _error("invalid_arguments", "Arguments failed validation.", details=_validation_details(exc))
        try:
            quote = order_service.quote_order(self._db, self._restaurant.id, request.items)
        except UnprocessableError as exc:
            problems = [{"menu_item_id": e.get("input"), "problem": e["type"], "message": e["msg"]}
                        | ({"available_quantity": e["available"]} if "available" in e else {})
                        for e in exc.detail]  # fmt: skip
            excluded = {line.menu_item_id for line in request.items}
            return _error(
                "items_not_orderable",
                "Some items cannot be ordered as requested. Tell the customer why, and offer only "
                "the alternatives listed here (they are in stock now).",
                problems=problems,
                alternatives=self._alternatives(excluded),
            )
        except ConflictError as exc:
            return _error("cannot_order", exc.detail)

        fields = {
            "customer_name": request.customer_name,
            "customer_phone": request.customer_phone,
            "notes": request.notes,
            "quote": quote,
        }
        pending = self._session.pending
        if pending is None or not pending.same_request(**fields):
            pending = PendingOrder.new(**fields)
            self._session.pending = pending
        return {
            "status": "awaiting_customer_confirmation",
            "order_placed": False,
            "proposal": proposal_result(pending, self._restaurant.currency),
            "next_step": (
                "No order has been placed and no stock is reserved. Read these items and the total "
                "to the customer, ask whether they want to add, remove or change anything, and ask "
                "them to confirm. The order is placed only when the customer confirms."
            ),
        }

    def _alternatives(self, excluded: set[uuid.UUID]) -> list[dict[str, Any]]:
        items, _ = menu_service.list_menu_items(
            self._db, self._restaurant.id, limit=MAX_ALTERNATIVES + len(excluded), offset=0,
            is_available=True, in_stock=True,
        )  # fmt: skip
        currency = self._restaurant.currency
        return [
            {"menu_item_id": str(i.id), "name": i.name, "price": format_money(i.price_paise, currency),
             "available_quantity": i.available_quantity}
            for i in items if i.id not in excluded
        ][:MAX_ALTERNATIVES]  # fmt: skip

    def _get_order_status(self, args: GetOrderStatusArgs) -> dict[str, Any]:
        orders = []
        for order_id in self._session.confirmed.values():
            try:
                order = order_service.get_order(self._db, order_id)
            except NotFoundError:
                continue
            if args.order_number is None or order.order_number == args.order_number:
                orders.append(order)
        if not orders:
            return _error("no_orders", "No order with that number was placed in this conversation.")
        currency = self._restaurant.currency
        results = []
        for order in orders:
            payment = order.payment
            self.verified_claims.add("order_placed")
            if payment is not None and payment.status == "succeeded":
                self.verified_claims.add("payment")
            results.append({
                "order_number": order.order_number,
                "status": order.status,
                "cancel_reason": order.cancel_reason,
                "items": [{"name": line.item_name, "quantity": line.quantity} for line in order.items],
                "total": format_money(order.total_paise, currency),
                "reservation_expires_at": (order.reservation_expires_at.isoformat()
                                           if order.reservation_expires_at else None),
                "payment_status": payment.status if payment is not None else None,
                "payment_mode": "demo (simulated, no real money)",
            })  # fmt: skip
        self.order_statuses = results
        return {"status": "ok", "orders": results, "payment_notice": DEMO_NOTICE}
