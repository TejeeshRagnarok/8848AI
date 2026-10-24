"""FastAPI application entry point.

Run locally with:
    python -m uvicorn app.main:app --reload
"""

import asyncio
import logging
from pathlib import Path
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.config import settings
from app.database import DatabaseUnavailableError, get_db, get_session_factory
from app.agent.llm import LLMNotConfiguredError, LLMResponseError, LLMUnavailableError
from app.routers import agent, dashboard, inventory, menu_items, orders, payments, restaurants
from app.services.orders import expire_reservations
from app.services.common import (
    ConflictError,
    NotFoundError,
    UnprocessableError,
    constraint_message,
)
from app.telephony import exotel

logger = logging.getLogger(__name__)

# Uvicorn only configures its own loggers, so the application's INFO logs
# (call starts, orders, payments) were silently dropped.  Messages never
# contain secrets or full phone numbers.
_app_logger = logging.getLogger("app")
if not _app_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
    _app_logger.addHandler(_handler)
    _app_logger.setLevel(logging.INFO)


def _sweep_once() -> None:
    with get_session_factory()() as session:
        expire_reservations(session)


async def _reservation_sweeper(interval: int) -> None:
    """Expire lapsed reservations every ``interval`` seconds (in a thread, so
    database work never blocks the event loop)."""
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(_sweep_once)
        except DatabaseUnavailableError:
            pass  # no database configured; nothing to sweep
        except Exception:
            logger.exception("Reservation sweep failed; will retry")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    task = None
    if settings.reservation_sweep_seconds > 0 and settings.database_url:
        task = asyncio.create_task(_reservation_sweeper(settings.reservation_sweep_seconds))
    try:
        yield
    finally:
        if task is not None:
            task.cancel()


app = FastAPI(
    lifespan=lifespan,
    title=settings.app_name,
    description=(
        "AI-powered voice ordering agent for restaurants. "
        "Handles inbound calls, menu queries, and order management.\n\n"
        "**DEMO PAYMENT MODE:** payments are simulated; no real money is ever charged.\n\n"
        "Staff endpoints need the `X-Staff-Key` header; a customer's own order needs its "
        "`X-Order-Token`. The agent chat API is unauthenticated and its sessions have no "
        "ownership checks: development only, do not expose this service publicly."
    ),
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)


# ── Exception handlers ────────────────────────────────────────────────────────


@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_handler(
    request: Request, exc: DatabaseUnavailableError
) -> JSONResponse:
    """Return 503 when the database is unreachable or not configured."""
    logger.warning("Database unavailable: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"status": "error", "database": "unreachable"},
    )


@app.exception_handler(OperationalError)
async def database_operational_error_handler(
    request: Request, exc: OperationalError
) -> JSONResponse:
    """Treat connection failures during a request like an unavailable database."""
    logger.warning("Database operational error: %s", type(exc.orig).__name__)
    return JSONResponse(
        status_code=503,
        content={"status": "error", "database": "unreachable"},
    )


@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": exc.detail})


@app.exception_handler(ConflictError)
async def conflict_handler(request: Request, exc: ConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": exc.detail})


@app.exception_handler(UnprocessableError)
async def unprocessable_handler(request: Request, exc: UnprocessableError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.detail})


@app.exception_handler(LLMNotConfiguredError)
@app.exception_handler(LLMUnavailableError)
async def llm_unavailable_handler(request: Request, exc: LLMNotConfiguredError) -> JSONResponse:
    """Assistant missing or provider down: 503 with a fixed, safe message."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMResponseError)
async def llm_response_handler(request: Request, exc: LLMResponseError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(IntegrityError)
async def integrity_error_handler(request: Request, exc: IntegrityError) -> JSONResponse:
    """Safety net for constraint violations not caught by a service.

    The request's session is rolled back when ``get_db`` closes it.
    """
    return JSONResponse(status_code=409, content={"detail": constraint_message(exc)})


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(restaurants.router)
app.include_router(menu_items.router)
app.include_router(orders.router)
app.include_router(inventory.router)
app.include_router(payments.router)
app.include_router(agent.router)
app.include_router(exotel.router)
app.include_router(dashboard.router)


# ── Dashboard (static single-page app; data via /api/v1/dashboard) ────────────

DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
app.mount("/dashboard/static", StaticFiles(directory=DASHBOARD_DIR), name="dashboard-static")

_DASHBOARD_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


@app.get("/dashboard", include_in_schema=False)
def dashboard_page() -> FileResponse:
    """The staff dashboard.  The page itself holds no data; everything is
    loaded after sign-in from the authenticated dashboard API."""
    return FileResponse(DASHBOARD_DIR / "index.html", headers=_DASHBOARD_HEADERS)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe — returns 200 OK when the application is running.

    Does **not** check database connectivity; use ``/health/db`` for that.
    """
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


@app.get("/health/db", tags=["ops"])
def health_db(db: Session = Depends(get_db)) -> JSONResponse:
    """Database readiness probe.

    Executes ``SELECT 1`` against PostgreSQL and returns 200 on success.
    Returns 503 if the database is unreachable, without leaking credentials.
    """
    try:
        db.execute(text("SELECT 1"))
        return JSONResponse(
            status_code=200,
            content={"status": "ok", "database": "connected"},
        )
    except Exception:
        logger.exception("Database health check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "database": "unreachable"},
        )

