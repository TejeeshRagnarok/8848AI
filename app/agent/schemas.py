"""Request/response schemas for the agent chat endpoint."""

import uuid
from typing import Annotated

from pydantic import BaseModel, StringConstraints, model_validator

from app.schemas import OrderRead
from app.schemas.common import RequestSchema

MAX_MESSAGE_CHARS = 1000

Message = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS)
]
SessionId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{20,64}$")]
ProposalId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]


class AgentChatRequest(RequestSchema):
    """One customer turn.

    - ``message``: what the customer said.
    - ``session_id``: from a previous response; omit to start a conversation.
    - ``confirm_proposal_id``: set only when the customer explicitly confirms
      the pending order shown in a previous response.  This is the ONLY way an
      order gets placed; the model cannot set it.
    """

    message: Message | None = None
    session_id: SessionId | None = None
    confirm_proposal_id: ProposalId | None = None

    @model_validator(mode="after")
    def _message_or_confirmation(self):
        if self.message is None and self.confirm_proposal_id is None:
            raise ValueError("provide a message, a confirm_proposal_id, or both")
        if self.confirm_proposal_id is not None and self.session_id is None:
            raise ValueError("confirm_proposal_id requires the session_id it belongs to")
        return self


class PendingOrderLine(BaseModel):
    menu_item_id: uuid.UUID
    name: str
    quantity: int
    unit_price_paise: int
    line_total_paise: int


class PendingOrderSummary(BaseModel):
    """Server-computed proposal awaiting confirmation (not an order yet)."""

    proposal_id: str
    items: list[PendingOrderLine]
    subtotal_paise: int
    total_paise: int
    total: str
    currency: str
    customer_name: str | None
    customer_phone: str
    notes: str | None


class AgentChatResponse(BaseModel):
    reply: str
    session_id: str
    # True while a proposal awaits confirmation; send its proposal_id back as
    # confirm_proposal_id once the customer agrees.
    requires_confirmation: bool
    pending_order: PendingOrderSummary | None = None
    # Set only when an order was actually created (or an earlier confirmation
    # of the same proposal was retried; then ``order_replayed`` is true).
    order: OrderRead | None = None
    order_replayed: bool = False
