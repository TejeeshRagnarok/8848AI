"""DEMO payment endpoints (staff only).  No real money moves, ever."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.payment import DemoPaymentSimulation, PaymentRead
from app.security import require_staff
from app.services import payments as service

router = APIRouter(prefix="/api/v1", tags=["payments (demo)"], dependencies=[Depends(require_staff)])

SessionDep = Annotated[Session, Depends(get_db)]


@router.get("/orders/{order_id}/payment", response_model=PaymentRead)
def get_order_payment(order_id: uuid.UUID, db: SessionDep):
    """The order's payment record."""
    return service.get_order_payment(db, order_id)


@router.get("/payments/{payment_id}", response_model=PaymentRead)
def get_payment(payment_id: uuid.UUID, db: SessionDep):
    return service.get_payment(db, payment_id)


@router.post(
    "/payments/{payment_id}/simulate",
    response_model=PaymentRead,
    responses={200: {"description": "Simulated outcome applied (or already applied)."}},
)
def simulate_payment(
    payment_id: uuid.UUID, body: DemoPaymentSimulation, db: SessionDep, response: Response
):
    """**DEMO PAYMENT MODE.** Simulate the gateway's answer for a pending payment.

    - ``succeeded``: the order becomes ``confirmed`` and its reserved stock is sold.
    - ``failed``: the order is cancelled and its reserved stock released.

    ``amount_paise`` must equal the payment amount (422 otherwise).  Repeating
    the same outcome returns the payment with ``Idempotent-Replayed: true`` and
    no side effects.  409 if the payment is already final with another
    outcome, or the reservation expired (the order is then cancelled).
    """
    result = service.simulate(db, payment_id, body.outcome, body.amount_paise, reason=body.reason)
    if result.replayed:
        response.headers["Idempotent-Replayed"] = "true"
    response.status_code = status.HTTP_200_OK
    return result.payment
