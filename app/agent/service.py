"""The restaurant ordering agent.

Two kinds of turn:

1. **Conversation** (``message`` only): a bounded loop of model calls and tool
   executions.  Tools can read the menu and *propose* an order; nothing is
   ever placed in this kind of turn.
2. **Confirmation** (``confirm_proposal_id`` set by the client when the
   customer explicitly agrees): handled entirely by the server, without the
   model.  The stored proposal is placed through the order engine with its
   stable idempotency key.  The engine re-prices it under row locks in the
   same transaction and refuses if anything changed, in which case the
   customer is shown the updated proposal and asked to confirm again.

All model-written replies pass through a guard that blocks false claims
(order placed, payment, delivery, messaging).
"""

import json
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.agent.llm import ChatProvider
from app.agent.prompts import (
    MENU_CONTEXT_MAX_ITEMS,
    caller_context,
    live_context,
    system_prompt,
    unsupported_claim,
)
from app.agent.schemas import AgentChatRequest, AgentChatResponse, PendingOrderSummary
from app.agent.sessions import AgentSession, PendingOrder, SessionStore
from app.agent.tools import TOOL_DEFINITIONS, ToolExecutor, format_money, proposal_result
from app.config import Settings, settings
from app.models import Order, OrderSource, Restaurant
from app.schemas import OrderCreate, OrderRead
from app.services import menu_items as menu_service
from app.services import orders as order_service
from app.services.common import ConflictError, NotFoundError, UnprocessableError
from app.services.orders import QuoteChangedError
from app.services.restaurants import get_restaurant

logger = logging.getLogger(__name__)

LIMIT_REPLY = (
    "Sorry, I couldn't finish that request. Could you tell me again, a little more simply, "
    "what you'd like?"
)


def load_menu_context(db: Session, restaurant: Restaurant) -> list | None:
    """The whole menu, fresh from the database, if small enough to send with
    every turn (one query; saves a get_menu round-trip to the model)."""
    items, total = menu_service.list_menu_items(
        db, restaurant.id, limit=MENU_CONTEXT_MAX_ITEMS, offset=0
    )
    return items if total <= MENU_CONTEXT_MAX_ITEMS else None


class RestaurantAgent:
    def __init__(
        self,
        db: Session,
        store: SessionStore,
        provider_factory: Callable[[], ChatProvider],
        config: Settings = settings,
        executor_factory: Callable[..., ToolExecutor] = ToolExecutor,
    ) -> None:
        self._db = db
        self._store = store
        self._provider_factory = provider_factory
        self._config = config
        self._executor_factory = executor_factory

    def chat(
        self,
        restaurant_id: uuid.UUID,
        request: AgentChatRequest,
        *,
        caller_phone: str | None = None,
        source: OrderSource = OrderSource.AI_CHAT,
    ) -> AgentChatResponse:
        """``caller_phone`` is the number a phone call came from (set by the
        telephony layer, never by the customer); the model is told it so it can
        offer it instead of asking."""
        restaurant = get_restaurant(self._db, restaurant_id)  # 404 if unknown
        if not restaurant.is_active:
            raise ConflictError("Restaurant is inactive; the ordering assistant is unavailable.")

        # Fail fast (503) before creating a session if the model is needed but
        # not configured.  Confirmations never need the model.
        provider = None if request.confirm_proposal_id else self._provider_factory()

        if request.session_id is not None:
            session = self._store.get(restaurant_id, request.session_id)
            if session is None:
                raise NotFoundError("Session not found or expired.")
        else:
            session = self._store.create(restaurant_id)

        if not session.lock.acquire(blocking=False):
            raise ConflictError("This conversation is already handling a message; please retry.")
        try:
            if request.confirm_proposal_id is not None:
                return self._confirm(restaurant, session, request, source)
            return self._converse(restaurant, session, request.message, provider, caller_phone)
        finally:
            session.lock.release()

    # ── Conversation turns ───────────────────────────────────────────────────

    def _converse(
        self,
        restaurant: Restaurant,
        session: AgentSession,
        message: str,
        provider: ChatProvider,
        caller_phone: str | None = None,
    ) -> AgentChatResponse:
        executor = self._executor_factory(self._db, restaurant, session)
        user_message = {"role": "user", "content": message}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(restaurant)},
            {"role": "system", "content": live_context(restaurant, load_menu_context(self._db, restaurant), session.pending)},
        ]
        if caller_phone:
            messages.append({"role": "system", "content": caller_context(caller_phone)})
        messages += [*session.history(), user_message]
        model_calls = tool_calls = 0
        reply = LIMIT_REPLY

        while model_calls < self._config.agent_max_model_calls:
            response = provider.complete(messages, TOOL_DEFINITIONS)
            model_calls += 1
            if not response.tool_calls:
                reply = response.content.strip()
                break
            if tool_calls + len(response.tool_calls) > self._config.agent_max_tool_calls:
                logger.warning("Agent tool-call limit reached")
                break
            messages.append(response.as_message())
            for call in response.tool_calls:
                result = executor.execute(call.name, call.arguments)
                tool_calls += 1
                # Results go back as data in a tool message, never as instructions.
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
                messages.append(tool_message)
        else:
            logger.warning("Agent model-call limit reached")

        reply = self._guard(
            reply, session, restaurant, executor.verified_claims, executor.order_statuses
        )
        # History keeps only what was said.  Tool calls and results (menus,
        # carts) are not replayed: every turn gets a fresh database snapshot
        # instead, which keeps facts current and the prompt small.
        session.add_turn([user_message, {"role": "assistant", "content": reply}])
        logger.info("Agent turn: %d model calls, %d tool calls", model_calls, tool_calls)
        return self._response(session, restaurant, reply)

    def _guard(
        self,
        reply: str,
        session: AgentSession,
        restaurant: Restaurant,
        verified: set[str] = frozenset(),
        statuses: list[dict[str, Any]] | None = None,
    ) -> str:
        claim = unsupported_claim(reply, verified)
        if claim is None:
            return reply
        logger.warning("Agent reply replaced: unsupported %s claim", claim)
        if statuses:
            # State the persisted facts the model just read, instead of its claim.
            facts = " ".join(
                f"According to our records, order {s['order_number']} is {s['status']} and its "
                f"demo payment is {s['payment_status'] or 'not set up'}."
                for s in statuses
            )
            return f"{facts} No real payment is taken in this demo."
        if session.pending is not None:
            return (
                f"Your order has not been placed yet. {self._describe(session.pending, restaurant)} "
                "Would you like me to place it?"
            )
        return "Sorry, I can't confirm that. How can I help with your order?"

    # ── Confirmation turns (server-only, no model) ───────────────────────────

    def _confirm(
        self,
        restaurant: Restaurant,
        session: AgentSession,
        request: AgentChatRequest,
        source: OrderSource = OrderSource.AI_CHAT,
    ) -> AgentChatResponse:
        proposal_id = request.confirm_proposal_id
        turn: list[dict[str, Any]] = []
        if request.message:
            turn.append({"role": "user", "content": request.message})

        # A retry of a confirmation that already succeeded: return that order.
        if proposal_id in session.confirmed:
            order = order_service.get_order(self._db, session.confirmed[proposal_id])
            reply = self._placed_reply(order, restaurant, again=True)
            return self._finish(session, restaurant, turn, reply, order=order, replayed=True)

        pending = session.pending
        if pending is None or pending.proposal_id != proposal_id:
            raise ConflictError("There is no pending order with that proposal_id to confirm.")
        if pending.customer_phone is None:
            # Nothing is placed without a contact number; the cart is kept.
            reply = (
                f"{self._describe(pending, restaurant)} Before I place it, which phone number "
                "should I put on the order?"
            )
            return self._finish(session, restaurant, turn, reply)

        data = OrderCreate(
            customer_name=pending.customer_name,
            customer_phone=pending.customer_phone,
            notes=pending.notes,
            items=[
                {"menu_item_id": line.menu_item_id, "quantity": line.quantity}
                for line in pending.quote.lines
            ],
            idempotency_key=pending.idempotency_key,
        )
        try:
            result = order_service.create_order(
                self._db, restaurant.id, data, expected=pending.quote, source=source
            )
        except QuoteChangedError as exc:
            session.pending = PendingOrder.new(
                customer_name=pending.customer_name,
                customer_phone=pending.customer_phone,
                notes=pending.notes,
                quote=exc.quote,
            )
            reply = self._changed_reply(pending, session.pending, restaurant)
            return self._finish(session, restaurant, turn, reply)
        except UnprocessableError as exc:
            session.pending = None
            problems = " ".join(e["msg"] for e in exc.detail)
            reply = (
                f"Sorry, I couldn't place the order. {problems} "
                "Would you like to change your order?"
            )
            return self._finish(session, restaurant, turn, reply)
        except ConflictError as exc:
            session.pending = None
            reply = f"Sorry, I couldn't place the order. {exc.detail}"
            return self._finish(session, restaurant, turn, reply)
        # Any other failure (e.g. database unavailable) propagates as an HTTP
        # error; the proposal is kept so the confirmation can be retried
        # safely with the same idempotency key.

        session.pending = None
        session.remember_confirmed(proposal_id, result.order.id)
        logger.info("Agent placed order %s", result.order.order_number)
        reply = self._placed_reply(result.order, restaurant, again=not result.created)
        return self._finish(
            session, restaurant, turn, reply, order=result.order, replayed=not result.created
        )

    # ── Replies and responses ────────────────────────────────────────────────

    @staticmethod
    def _describe(pending: PendingOrder, restaurant: Restaurant) -> str:
        items = ", ".join(f"{line.quantity} x {line.item_name}" for line in pending.quote.lines)
        total = format_money(pending.quote.total_paise, restaurant.currency)
        return f"The order is {items}, for a total of {total}."

    def _changed_reply(self, old: PendingOrder, new: PendingOrder, restaurant: Restaurant) -> str:
        before = {line.menu_item_id: line for line in old.quote.lines}
        changes = []
        for line in new.quote.lines:
            prev = before.get(line.menu_item_id)
            if prev is not None and prev.unit_price_paise != line.unit_price_paise:
                changes.append(
                    f"{line.item_name} is now {format_money(line.unit_price_paise, restaurant.currency)} "
                    f"(was {format_money(prev.unit_price_paise, restaurant.currency)})."
                )
            elif prev is not None and prev.item_name != line.item_name:
                changes.append(f"{prev.item_name} is now called {line.item_name}.")
        return (
            "Before placing your order: something on the menu has changed. "
            f"{' '.join(changes)} {self._describe(new, restaurant)} "
            "Would you like me to place the order with these details?"
        ).replace("  ", " ")

    @staticmethod
    def _placed_reply(order: Order, restaurant: Restaurant, *, again: bool) -> str:
        """Server-written (never model-written) placement message, from the DB row."""
        total = format_money(order.total_paise, restaurant.currency)
        prefix = "That order was already placed." if again else "Thank you, your order has been placed."
        reply = f"{prefix} Your order number is {order.order_number} and the total is {total}."
        if order.status == "pending":
            held = ""
            if order.reservation_expires_at is not None:
                left = (order.reservation_expires_at - datetime.now(UTC)).total_seconds()
                held = f" Your items are reserved for {max(1, round(left / 60))} minutes."
            reply += (
                f"{held} This is a demo: no real payment is taken. The demo payment is pending, and "
                "the order is confirmed once that payment succeeds. Ask me for your order status "
                "at any time."
            )
        return reply

    def _finish(
        self,
        session: AgentSession,
        restaurant: Restaurant,
        turn: list[dict[str, Any]],
        reply: str,
        *,
        order: Order | None = None,
        replayed: bool = False,
    ) -> AgentChatResponse:
        turn.append({"role": "assistant", "content": reply})
        session.add_turn(turn)
        return self._response(session, restaurant, reply, order=order, replayed=replayed)

    @staticmethod
    def _response(
        session: AgentSession,
        restaurant: Restaurant,
        reply: str,
        *,
        order: Order | None = None,
        replayed: bool = False,
    ) -> AgentChatResponse:
        pending = session.pending
        return AgentChatResponse(
            reply=reply,
            session_id=session.id,
            requires_confirmation=pending is not None,
            pending_order=(
                PendingOrderSummary(**proposal_result(pending, restaurant.currency))
                if pending is not None
                else None
            ),
            order=OrderRead.model_validate(order) if order is not None else None,
            order_replayed=replayed,
        )
