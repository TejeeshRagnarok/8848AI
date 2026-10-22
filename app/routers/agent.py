"""Agent chat endpoint (development only — no authentication yet).

Not safe to expose publicly: there is no caller authentication, no
authorization, and no check that a session belongs to the caller beyond
knowing its random ID.
"""

import uuid
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.agent.llm import ChatProvider, get_provider
from app.agent.schemas import AgentChatRequest, AgentChatResponse
from app.agent.service import RestaurantAgent
from app.agent.sessions import InMemorySessionStore, SessionStore
from app.config import settings
from app.database import get_db

router = APIRouter(prefix="/api/v1", tags=["agent"])

_session_store = InMemorySessionStore(
    ttl_seconds=settings.agent_session_ttl_seconds, max_sessions=settings.agent_max_sessions
)


def get_session_store() -> SessionStore:
    return _session_store


def get_provider_factory() -> Callable[[], ChatProvider]:
    """Return a factory so the provider is only built (and required) when the
    model is actually needed."""
    return get_provider


@router.post("/restaurants/{restaurant_id}/agent/chat", response_model=AgentChatResponse)
def chat(
    restaurant_id: uuid.UUID,
    body: AgentChatRequest,
    db: Annotated[Session, Depends(get_db)],
    store: Annotated[SessionStore, Depends(get_session_store)],
    provider_factory: Annotated[Callable[[], ChatProvider], Depends(get_provider_factory)],
):
    """Send one customer message to the ordering assistant.

    Start a conversation without ``session_id``; reuse the returned one after.
    When ``requires_confirmation`` is true, show ``pending_order`` to the
    customer and, only if they explicitly agree, send
    ``confirm_proposal_id = pending_order.proposal_id``.  That is the only way
    an order is placed.

    Errors: 404 unknown restaurant or session · 409 inactive restaurant, busy
    session, or nothing matching to confirm · 422 invalid body · 502 invalid
    model response · 503 assistant not configured or provider unavailable.
    """
    return RestaurantAgent(db, store, provider_factory).chat(restaurant_id, body)
