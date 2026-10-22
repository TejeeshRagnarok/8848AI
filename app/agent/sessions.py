"""Agent conversation sessions and server-side pending orders.

DEVELOPMENT-ONLY STORAGE.  ``InMemorySessionStore`` keeps sessions in this
process's memory:

- sessions are lost on restart and are not shared between worker processes
  (run a single worker, or replace the store before scaling out);
- the store is bounded (``max_sessions``, least-recently-used evicted first)
  and sessions expire after ``ttl_seconds`` of inactivity;
- session IDs are random server-generated tokens bound to one restaurant, but
  there is no caller authentication, so anyone holding an ID can use it.

The agent depends only on the ``SessionStore`` protocol, so a durable store
(e.g. PostgreSQL or Redis) can replace this without touching agent logic.
"""

import secrets
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.services.orders import OrderQuote

MAX_HISTORY_TURNS = 8
MAX_CONFIRMED_REMEMBERED = 10


@dataclass(frozen=True)
class PendingOrder:
    """An order proposal awaiting explicit customer confirmation.

    Everything needed to place the order lives here, server-side; the model
    cannot alter it after the fact.  ``idempotency_key`` is stable for this
    proposal, so retried confirmations can never create a second order.
    """

    proposal_id: str
    idempotency_key: str
    customer_name: str | None
    customer_phone: str
    notes: str | None
    quote: OrderQuote

    @classmethod
    def new(cls, *, customer_name, customer_phone, notes, quote: OrderQuote) -> "PendingOrder":
        proposal_id = uuid.uuid4().hex
        return cls(
            proposal_id=proposal_id,
            idempotency_key=f"agent-{proposal_id}",
            customer_name=customer_name,
            customer_phone=customer_phone,
            notes=notes,
            quote=quote,
        )

    def same_request(self, *, customer_name, customer_phone, notes, quote: OrderQuote) -> bool:
        return (self.customer_name, self.customer_phone, self.notes, self.quote) == (
            customer_name, customer_phone, notes, quote,
        )  # fmt: skip


@dataclass
class AgentSession:
    id: str
    restaurant_id: uuid.UUID
    last_used: float
    # Each turn is the list of chat messages it produced (user, assistant,
    # tool calls/results); trimmed by whole turns so tool pairs stay intact.
    turns: list[list[dict[str, Any]]] = field(default_factory=list)
    pending: PendingOrder | None = None
    # proposal_id -> order_id for recently confirmed proposals (safe retries).
    confirmed: OrderedDict[str, uuid.UUID] = field(default_factory=OrderedDict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def history(self) -> list[dict[str, Any]]:
        return [message for turn in self.turns for message in turn]

    def add_turn(self, messages: list[dict[str, Any]]) -> None:
        self.turns.append(messages)
        del self.turns[:-MAX_HISTORY_TURNS]

    def remember_confirmed(self, proposal_id: str, order_id: uuid.UUID) -> None:
        self.confirmed[proposal_id] = order_id
        while len(self.confirmed) > MAX_CONFIRMED_REMEMBERED:
            self.confirmed.popitem(last=False)


class SessionStore(Protocol):
    def get(self, restaurant_id: uuid.UUID, session_id: str) -> AgentSession | None:
        """The live session, or None if unknown, expired, or another restaurant's."""
        ...

    def create(self, restaurant_id: uuid.UUID) -> AgentSession: ...


class InMemorySessionStore:
    """Bounded, TTL-based, thread-safe, single-process session store."""

    def __init__(self, *, ttl_seconds: int, max_sessions: int, clock=time.monotonic) -> None:
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._clock = clock
        self._sessions: OrderedDict[str, AgentSession] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, restaurant_id: uuid.UUID, session_id: str) -> AgentSession | None:
        with self._lock:
            self._expire()
            session = self._sessions.get(session_id)
            if session is None or session.restaurant_id != restaurant_id:
                return None
            session.last_used = self._clock()
            self._sessions.move_to_end(session_id)
            return session

    def create(self, restaurant_id: uuid.UUID) -> AgentSession:
        with self._lock:
            self._expire()
            while len(self._sessions) >= self._max:
                self._sessions.popitem(last=False)  # evict least recently used
            session = AgentSession(
                id=secrets.token_urlsafe(24), restaurant_id=restaurant_id, last_used=self._clock()
            )
            self._sessions[session.id] = session
            return session

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _expire(self) -> None:
        cutoff = self._clock() - self._ttl
        while self._sessions:
            oldest = next(iter(self._sessions.values()))
            if oldest.last_used >= cutoff:
                break
            self._sessions.popitem(last=False)
