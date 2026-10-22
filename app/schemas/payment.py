"""Schemas for DEMO payments.  Every response carries the demo notice."""

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.models.payment import DEMO_NOTICE, DemoOutcome
from app.schemas.common import RequestSchema


class PaymentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    order_id: uuid.UUID
    amount_paise: int
    currency: str
    mode: str
    status: str
    failure_reason: str | None
    succeeded_at: datetime | None
    closed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    simulated: bool = True
    notice: str = DEMO_NOTICE


class DemoPaymentSimulation(RequestSchema):
    """Simulate the payment gateway's answer (staff only)."""

    outcome: DemoOutcome
    # Must equal the payment amount; the server never takes the client's word
    # for what is owed.
    amount_paise: Annotated[int, Field(strict=True, gt=0)]
    reason: Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)] | None = None
