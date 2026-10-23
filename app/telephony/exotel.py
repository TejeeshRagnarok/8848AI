"""Exotel Voicebot applet: phone calls into the ordering agent.

Exotel opens a WebSocket per call and exchanges JSON messages:

- from Exotel: ``connected``, ``start`` (call details, custom parameters),
  ``media`` (base64 16-bit mono PCM), ``dtmf``, ``mark``, ``stop``;
- to Exotel: ``media`` (audio to play), ``mark`` (tell me when it finished),
  ``clear`` (drop queued audio, for barge-in).

Per call: voice activity detection cuts the caller's audio into utterances,
each is transcribed, sent to ``RestaurantAgent`` as one chat turn, and the
reply is synthesised and streamed back.  Orders are still placed only by the
server-side confirmation path: when a proposal is pending, a clear "yes" (or
pressing 1) sends ``confirm_proposal_id``; the model never places an order.

Development only, like the chat API: protect the socket with
``EXOTEL_WS_USERNAME`` / ``EXOTEL_WS_PASSWORD`` before exposing it publicly.
"""

import asyncio
import base64
import binascii
import json
import logging
import re
import secrets
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Annotated, Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session

from app.agent.llm import ChatProvider, LLMError
from app.agent.schemas import MAX_MESSAGE_CHARS, AgentChatRequest
from app.agent.service import RestaurantAgent
from app.agent.sessions import SessionStore
from app.config import Settings, settings
from app.database import get_session_factory
from app.routers.agent import get_provider_factory, get_session_store
from app.services.common import ConflictError, NotFoundError
from app.services.restaurants import get_restaurant
from app.telephony.audio import UtteranceDetector, exotel_chunks
from app.telephony.speech import SpeechError, SpeechNotConfiguredError, get_speech, speakable

logger = logging.getLogger(__name__)

router = APIRouter(tags=["telephony"])

WS_PATH = "/telephony/exotel/voicebot"

CONFIRM_HINT = "To place this order, say yes or press 1."
SORRY_UNAVAILABLE = (
    "Sorry, our ordering assistant is unavailable right now. Please call again later. Goodbye."
)
SORRY_NOT_HEARD = "Sorry, I didn't catch that. Could you say it again?"
SORRY_CLOSED = "Sorry, this restaurant is not taking phone orders right now. Goodbye."

DbFactory = Callable[[], AbstractContextManager[Session]]


# ── Dependencies (overridable in tests) ──────────────────────────────────────


@contextmanager
def _open_db() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def get_db_factory() -> DbFactory:
    return _open_db


def get_speech_factory() -> Callable[[], Any]:
    return get_speech


def get_settings() -> Settings:
    return settings


# ── Confirmation by voice ────────────────────────────────────────────────────

_AFFIRMATIVE = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed", "correct",
    "haan", "ha", "han", "ji", "please", "go", "ahead", "place", "it", "the", "order", "that's",
    "thats", "right", "do", "that", "fine", "alright", "absolutely", "definitely", "of", "course",
}  # fmt: skip
_AFFIRMATIVE_CORE = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed", "correct",
    "haan", "ha", "han", "ji", "alright", "absolutely", "definitely", "place", "go", "right",
}  # fmt: skip


def is_affirmative(text: str) -> bool:
    """True only for a short, unambiguous agreement ("yes", "yes please,
    place it").  Anything else (changes, questions, "no") goes to the model."""
    words = re.findall(r"[a-z']+", text.lower())
    if not words or len(words) > 6:
        return False
    return all(w in _AFFIRMATIVE for w in words) and any(w in _AFFIRMATIVE_CORE for w in words)


# ── Caller ID ────────────────────────────────────────────────────────────────

# Same shape the order schema accepts for customer_phone (5-20 characters).
_CALLER_ID = re.compile(r"^\+?[0-9][0-9 -]{3,18}[0-9]$")


def normalize_caller(raw: Any) -> str | None:
    """The caller ID Exotel reported, if it looks like a phone number.

    Withheld, unknown or malformed values become None, so the agent asks for
    a contact number instead.  Every caller Exotel connects is served either
    way: the number only pre-fills the order's contact number and is never
    treated as proof of identity.
    """
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    return value if _CALLER_ID.match(value) else None


def mask_phone(phone: str | None) -> str:
    """Last four digits only, for logs."""
    digits = re.sub(r"\D", "", phone or "")
    return f"***{digits[-4:]}" if len(digits) >= 4 else "unknown"


# ── Auth ─────────────────────────────────────────────────────────────────────


def _authorized(websocket: WebSocket, config: Settings) -> bool:
    if not config.exotel_ws_username or config.exotel_ws_password is None:
        return True
    header = websocket.headers.get("authorization", "")
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        supplied = base64.b64decode(encoded, validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return False
    expected = f"{config.exotel_ws_username}:{config.exotel_ws_password.get_secret_value()}"
    return secrets.compare_digest(supplied.encode(), expected.encode())


# ── One call ─────────────────────────────────────────────────────────────────


class ExotelCall:
    """State and behaviour for one phone call (one WebSocket)."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        stream_sid: str,
        call_sid: str,
        restaurant_id: uuid.UUID,
        caller: str | None,
        speech: Any,
        store: SessionStore,
        provider_factory: Callable[[], ChatProvider],
        db_factory: DbFactory,
        config: Settings,
    ) -> None:
        self.ws = websocket
        self.stream_sid = stream_sid
        self.call_sid = call_sid
        self.restaurant_id = restaurant_id
        self.caller = caller
        self.speech = speech
        self.store = store
        self.provider_factory = provider_factory
        self.db_factory = db_factory
        self.config = config
        self.rate = config.exotel_sample_rate
        self.detector = UtteranceDetector(
            sample_rate=self.rate,
            energy_threshold=config.vad_energy_threshold,
            silence_ms=config.vad_silence_ms,
            min_speech_ms=config.vad_min_speech_ms,
            max_utterance_ms=config.vad_max_utterance_ms,
        )
        self.queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self.session_id: str | None = None
        self.pending_proposal: str | None = None
        self.playing = False
        self.hangup = False
        self._marks = 0
        self._last_mark: str | None = None
        self._send_lock = asyncio.Lock()

    # ── Sending ──────────────────────────────────────────────────────────────

    async def send(self, message: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(message))

    async def speak(self, text: str) -> None:
        try:
            pcm = await asyncio.to_thread(self.speech.synthesize, speakable(text), self.rate)
        except SpeechError as exc:
            logger.warning("Call %s: speech synthesis failed (%s)", self.call_sid, type(exc).__name__)
            return
        for chunk in exotel_chunks(pcm):
            await self.send({
                "event": "media",
                "stream_sid": self.stream_sid,
                "media": {"payload": base64.b64encode(chunk).decode()},
            })  # fmt: skip
        self._marks += 1
        self._last_mark = f"reply-{self._marks}"
        self.playing = True
        await self.send({"event": "mark", "stream_sid": self.stream_sid,
                         "mark": {"name": self._last_mark}})  # fmt: skip

    # ── Inbound events (called from the receive loop) ────────────────────────

    async def on_media(self, payload: str) -> None:
        try:
            pcm = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return
        for event in self.detector.feed(pcm):
            if event.kind == "speech_start" and self.playing:
                # Barge-in: the caller talks over the bot, stop playback.
                self.playing = False
                await self.send({"event": "clear", "stream_sid": self.stream_sid})
            elif event.kind == "utterance":
                self.queue.put_nowait(("speech", event.audio))

    def on_mark(self, name: str | None) -> None:
        if name is not None and name == self._last_mark:
            self.playing = False

    def on_dtmf(self, digit: str | None) -> None:
        if digit:
            self.queue.put_nowait(("dtmf", digit))

    # ── Worker: one item at a time, so turns never overlap ───────────────────

    async def run(self) -> None:
        if not await self._greet():
            self.hangup = True
            return
        while True:
            kind, value = await self.queue.get()
            if kind == "speech":
                await self._handle_speech(value)
            elif kind == "dtmf":
                await self._handle_dtmf(value)
            if self.hangup:
                return

    async def _greet(self) -> bool:
        try:
            restaurant = await asyncio.to_thread(self._load_restaurant)
        except Exception:
            logger.exception("Call %s: could not load restaurant", self.call_sid)
            restaurant = None
        if restaurant is None or not restaurant.is_active:
            await self.speak(SORRY_CLOSED)
            return False
        greeting = self.config.exotel_greeting or (
            f"Welcome to {restaurant.name}. I can tell you about the menu and take your order. "
            "What would you like?"
        )
        await self.speak(greeting)
        return True

    def _load_restaurant(self):
        with self.db_factory() as db:
            try:
                return get_restaurant(db, self.restaurant_id)
            except NotFoundError:
                return None

    async def _handle_speech(self, audio: bytes) -> None:
        try:
            text = await asyncio.to_thread(self.speech.transcribe, audio, self.rate)
        except SpeechError as exc:
            logger.warning("Call %s: transcription failed (%s)", self.call_sid, type(exc).__name__)
            await self.speak(SORRY_NOT_HEARD)
            return
        if not text:
            return
        confirm = self.pending_proposal is not None and is_affirmative(text)
        await self._turn(text[:MAX_MESSAGE_CHARS], confirm=confirm)

    async def _handle_dtmf(self, digit: str) -> None:
        if digit == "1" and self.pending_proposal is not None:
            await self._turn(None, confirm=True)

    async def _turn(self, text: str | None, *, confirm: bool) -> None:
        try:
            reply = await asyncio.to_thread(self._agent_turn, text, confirm)
        except LLMError as exc:
            logger.warning("Call %s: assistant unavailable (%s)", self.call_sid, type(exc).__name__)
            await self.speak(SORRY_UNAVAILABLE)
            self.hangup = True
            return
        except (ConflictError, NotFoundError) as exc:
            logger.info("Call %s: agent refused turn: %s", self.call_sid, exc.detail)
            reply = "Sorry, I couldn't do that. Could you tell me again what you'd like?"
        except Exception:
            logger.exception("Call %s: agent turn failed", self.call_sid)
            await self.speak(SORRY_UNAVAILABLE)
            self.hangup = True
            return
        await self.speak(reply)

    def _agent_turn(self, text: str | None, confirm: bool) -> str:
        request = AgentChatRequest(
            message=text,
            session_id=self.session_id,
            confirm_proposal_id=self.pending_proposal if confirm else None,
        )
        with self.db_factory() as db:
            agent = RestaurantAgent(db, self.store, self.provider_factory)
            response = agent.chat(self.restaurant_id, request, caller_phone=self.caller)
        self.session_id = response.session_id
        previous = self.pending_proposal
        self.pending_proposal = (
            response.pending_order.proposal_id if response.requires_confirmation else None
        )
        if response.order is not None:
            logger.info("Call %s placed order %s", self.call_sid, response.order.order_number)
        if self.pending_proposal is not None and self.pending_proposal != previous:
            return f"{response.reply} {CONFIRM_HINT}"
        return response.reply


# ── Endpoint ─────────────────────────────────────────────────────────────────


def _restaurant_id(custom: Any, config: Settings) -> uuid.UUID | None:
    raw = custom.get("restaurant_id") if isinstance(custom, dict) else None
    if raw:
        try:
            return uuid.UUID(str(raw))
        except ValueError:
            return None
    return config.exotel_restaurant_id


@router.websocket(WS_PATH)
async def exotel_voicebot(
    websocket: WebSocket,
    store: Annotated[SessionStore, Depends(get_session_store)],
    provider_factory: Annotated[Callable[[], ChatProvider], Depends(get_provider_factory)],
    speech_factory: Annotated[Callable[[], Any], Depends(get_speech_factory)],
    db_factory: Annotated[DbFactory, Depends(get_db_factory)],
    config: Annotated[Settings, Depends(get_settings)],
) -> None:
    """WebSocket for Exotel's Voicebot applet (bidirectional audio)."""
    if not _authorized(websocket, config):
        logger.warning("Exotel WebSocket rejected: bad or missing credentials")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    try:
        speech = speech_factory()
    except SpeechNotConfiguredError:
        logger.error("Exotel call rejected: speech services are not configured")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    await websocket.accept()
    call: ExotelCall | None = None
    worker: asyncio.Task | None = None
    try:
        while True:
            try:
                message = json.loads(await websocket.receive_text())
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            event = message.get("event")

            if event == "start" and call is None:
                start = message.get("start") or {}
                restaurant_id = _restaurant_id(start.get("custom_parameters"), config)
                if restaurant_id is None:
                    logger.error("Exotel call rejected: no valid restaurant_id configured")
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                    return
                call = ExotelCall(
                    websocket,
                    stream_sid=str(start.get("stream_sid") or message.get("stream_sid") or ""),
                    call_sid=str(start.get("call_sid") or "unknown"),
                    restaurant_id=restaurant_id,
                    caller=normalize_caller(start.get("from")),
                    speech=speech,
                    store=store,
                    provider_factory=provider_factory,
                    db_factory=db_factory,
                    config=config,
                )
                logger.info("Exotel call %s started (caller %s)", call.call_sid, mask_phone(call.caller))
                worker = asyncio.create_task(call.run())
            elif call is None:
                continue  # "connected", or anything before "start"
            elif event == "media":
                await call.on_media((message.get("media") or {}).get("payload") or "")
            elif event == "dtmf":
                call.on_dtmf((message.get("dtmf") or {}).get("digit"))
            elif event == "mark":
                call.on_mark((message.get("mark") or {}).get("name"))
            elif event == "stop":
                logger.info("Exotel call %s stopped (%s)", call.call_sid,
                            (message.get("stop") or {}).get("reason"))  # fmt: skip
                break

            if worker is not None and worker.done() and not call.playing:
                # The worker ended the conversation (e.g. assistant unavailable)
                # and its last words have finished playing.
                if not worker.cancelled() and worker.exception() is not None:
                    logger.error("Exotel call %s worker crashed", call.call_sid,
                                 exc_info=worker.exception())  # fmt: skip
                break
    except WebSocketDisconnect:
        return
    finally:
        if worker is not None and not worker.done():
            worker.cancel()
    try:
        await websocket.close()
    except RuntimeError:
        pass  # already closed
