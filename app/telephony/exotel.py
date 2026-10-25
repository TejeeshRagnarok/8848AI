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

Reliability rules (a caller must never be left in silence):

- every stage is bounded (provider HTTP timeouts, a per-turn deadline, a
  deadline for Exotel's ``start`` and for a stream that goes quiet);
- every turn ends in a logged terminal state: answered, recovered (a short
  spoken apology from cached audio) or the call is ended politely;
- fixed phrases (greeting, apologies, goodbye) are synthesised once and kept
  on disk (``VOICE_PHRASE_CACHE_DIR``), so they play even when the TTS
  provider is down or out of quota;
- repeated failures end the call with a spoken goodbye instead of silence.

Logs are structured per call and turn (stage, result, duration, error
category, audio bytes); they never contain phone numbers, keys, audio or what
the caller said.

Development only, like the chat API: protect the socket with
``EXOTEL_WS_USERNAME`` / ``EXOTEL_WS_PASSWORD`` before exposing it publicly.
"""

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import re
import secrets
import time
import uuid
import weakref
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session

from app.agent.llm import ChatProvider, LLMError
from app.agent.schemas import MAX_MESSAGE_CHARS, AgentChatRequest
from app.agent.service import RestaurantAgent
from app.agent.sessions import SessionStore
from app.config import Settings, settings
from app.database import get_session_factory
from app.models import OrderSource
from app.routers.agent import get_provider_factory, get_session_store
from app.services.common import ConflictError, NotFoundError
from app.services.restaurants import get_restaurant
from app.telephony.audio import SAMPLE_WIDTH, UtteranceDetector, exotel_chunks, silence
from app.telephony.speech import SpeechError, SpeechNotConfiguredError, get_speech, speakable

logger = logging.getLogger(__name__)

router = APIRouter(tags=["telephony"])

WS_PATH = "/telephony/exotel/voicebot"

CONFIRM_HINT = "To place this order, say yes or press 1."
SORRY_NOT_HEARD = "Sorry, I'm having trouble hearing you right now. Could you please repeat that?"
SORRY_RETRY = "Sorry, I'm having a little trouble. Could you say that again?"
SORRY_SPEECH = "Sorry, I'm having trouble speaking right now. Please say that again in a moment."
SORRY_UNAVAILABLE = (
    "Sorry, our ordering assistant is unavailable right now. Please call again later. Goodbye."
)
SORRY_CLOSED = "Sorry, this restaurant is not taking phone orders right now. Goodbye."
FILLER = "One moment."
STILL_WORKING = "Thanks for waiting, I'm still working on that."
# Fixed phrases kept ready (in memory and on disk) for when the TTS fails.
# Most critical first: they are what a caller hears when TTS is failing.
FIXED_PHRASES = (SORRY_SPEECH, SORRY_UNAVAILABLE, SORRY_NOT_HEARD, SORRY_RETRY, FILLER, SORRY_CLOSED,
                 STILL_WORKING)  # fmt: skip
MAX_CONSECUTIVE_FAILURES = 2  # then say goodbye and end the call
# Any of these will do when the preferred apology has no audio.
APOLOGIES = (SORRY_SPEECH, SORRY_RETRY, SORRY_NOT_HEARD)
MAX_MEDIA_BYTES = 64_000  # one inbound media frame; Exotel sends far less
GOODBYE_GRACE_SECONDS = 15  # max wait for the last words before closing
# Beyond the HTTP timeout, in case a provider call hangs below the HTTP layer.
STT_WATCHDOG_MARGIN = 2.0
TTS_WATCHDOG_MARGIN = 2.0
EMPTY_TRANSCRIPT_REPLY_MS = 1000  # this much speech transcribed to nothing gets a reply

DbFactory = Callable[[], AbstractContextManager[Session]]


def greeting_text(restaurant_name: str, config: Settings) -> str:
    return config.exotel_greeting or (
        f"Welcome to {restaurant_name}. I can tell you about the menu and take your order. "
        "What would you like?"
    )


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


# ── Fixed phrases (cached audio) ─────────────────────────────────────────────

# Memory cache per speech provider, backed by an optional disk cache so the
# phrases survive restarts and play even when the TTS provider is down.
_phrase_audio: "weakref.WeakKeyDictionary[Any, dict[tuple[int, str], bytes]]" = (
    weakref.WeakKeyDictionary()
)


def _phrase_file(directory: str, speech: Any, text: str, rate: int) -> Path:
    voice = getattr(speech, "voice_signature", type(speech).__name__)
    digest = hashlib.sha256(f"{voice}|{rate}|{speakable(text)}".encode()).hexdigest()[:32]
    return Path(directory) / f"{digest}.pcm"


def _read_phrase(path: Path) -> bytes | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return data if data and len(data) % SAMPLE_WIDTH == 0 else None


def _write_phrase(path: Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("Could not save phrase audio (%s)", type(exc).__name__)


async def cached_phrase(
    speech: Any, text: str, rate: int, directory: str | None = None, timeout: float | None = None
) -> bytes | None:
    """Audio for a fixed phrase: memory, then disk, then one synthesis
    (cut off after ``timeout`` seconds, if given)."""
    cache = _phrase_audio.setdefault(speech, {})
    key = (rate, text)
    if key in cache:
        return cache[key]
    path = _phrase_file(directory, speech, text, rate) if directory else None
    data = await asyncio.to_thread(_read_phrase, path) if path else None
    if data is None:
        try:
            data = await asyncio.wait_for(
                asyncio.to_thread(speech.synthesize, speakable(text), rate), timeout=timeout
            )
        except SpeechError as exc:
            logger.warning("Phrase synthesis failed (%s)", getattr(exc, "reason", type(exc).__name__))
            return None
        except TimeoutError:
            logger.warning("Phrase synthesis failed (timeout)")
            return None
        if not data:
            logger.warning("Phrase synthesis returned no audio")
            return None
        if path:
            await asyncio.to_thread(_write_phrase, path, data)
    cache[key] = data
    return data


async def prewarm_phrases(
    speech: Any, rate: int, directory: str | None, extra: Iterable[str] = ()
) -> tuple[int, int]:
    """Make sure every fixed phrase (and ``extra``, e.g. the greeting) has
    audio ready.  Returns (ready, failed).  Only missing phrases use TTS."""
    ready = failed = 0
    for phrase in (*extra, *FIXED_PHRASES):
        if await cached_phrase(speech, phrase, rate, directory) is None:
            failed += 1
        else:
            ready += 1
    return ready, failed


def speech_pieces(speech: Any, text: str) -> list[str]:
    """Pieces synthesised (and sent) one at a time, so playback of the first
    starts while the next is still being generated."""
    splitter = getattr(speech, "pieces", None)
    return splitter(text) if splitter else [text]


class CallEnded(Exception):
    """The WebSocket can no longer be written to; the call is over."""


# ── One call ─────────────────────────────────────────────────────────────────


class ExotelCall:
    """State and behaviour for one phone call (one WebSocket).

    Turn state: the receive loop only detects speech and queues utterances;
    one worker handles turns in order.  ``_utterances`` counts finished caller
    utterances: if it moves while a turn's reply is being prepared, the caller
    has said something new, so that reply is stale (see ``_handle_speech``).
    ``_generation`` changes on barge-in, which stops any reply audio still
    being sent.  Exotel's ``clear`` drops audio it has not started playing.

    ``_failures`` counts consecutive turns that could not be answered (speech
    recognition, assistant or speech synthesis failures).  Each one gets a
    spoken apology from cached audio; ``MAX_CONSECUTIVE_FAILURES`` in a row
    end the call with a spoken goodbye.  ``hangup`` makes the endpoint close
    the WebSocket once the last words have played.
    """

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
        self.end_reason: str | None = None
        self.turns = 0
        self.audio_bytes_sent = 0
        self.bad_frames = 0
        self.started_at = time.perf_counter()
        self._turn_no = 0
        self._utterances = 0
        self._generation = 0
        self._failures = 0  # consecutive turns that could not be answered
        self._marks = 0
        self._last_mark: str | None = None
        self._send_lock = asyncio.Lock()

    # ── Logging ──────────────────────────────────────────────────────────────

    def stage(
        self,
        stage: str,
        result: str,
        started: float | None = None,
        *,
        error: str | None = None,
        audio_bytes: int | None = None,
    ) -> None:
        """One structured line per stage: no content, no phone numbers."""
        parts = [f"call={self.call_sid}", f"turn={self._turn_no}", f"stage={stage}", f"result={result}"]
        if started is not None:
            parts.append(f"duration={time.perf_counter() - started:.2f}s")
        if error:
            parts.append(f"error={error}")
        if audio_bytes is not None:
            parts.append(f"audio_bytes={audio_bytes}")
        log = logger.info if result in ("ok", "skipped", "empty") else logger.warning
        log("voice %s", " ".join(parts))

    def end(self, reason: str) -> None:
        if self.end_reason is None:
            self.end_reason = reason
        self.hangup = True

    # ── Sending ──────────────────────────────────────────────────────────────

    async def send(self, message: dict[str, Any]) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_text(json.dumps(message))
            except (WebSocketDisconnect, RuntimeError, OSError) as exc:
                raise CallEnded(type(exc).__name__) from exc

    async def _send_audio(self, pcm: bytes, generation: int) -> int | None:
        """Queue audio with Exotel; bytes sent, or None if a barge-in cancelled it."""
        sent = 0
        for chunk in exotel_chunks(pcm, self.config.exotel_chunk_bytes):
            if generation != self._generation:
                return None
            await self.send({
                "event": "media",
                "stream_sid": self.stream_sid,
                "media": {"payload": base64.b64encode(chunk).decode()},
            })  # fmt: skip
            sent += len(chunk)
        self.playing = True
        self.audio_bytes_sent += sent
        return sent

    async def _mark(self) -> None:
        self._marks += 1
        self._last_mark = f"reply-{self._marks}"
        await self.send({"event": "mark", "stream_sid": self.stream_sid,
                         "mark": {"name": self._last_mark}})  # fmt: skip

    async def phrase(self, text: str) -> bool:
        """Play a fixed phrase from cached audio (synthesised once if needed)."""
        started = time.perf_counter()
        pcm = await cached_phrase(self.speech, text, self.rate, self.config.voice_phrase_cache_dir,
                                  timeout=self._tts_deadline())  # fmt: skip
        if pcm is None:
            self.stage("phrase", "failed", started, error="no_audio")
            return False
        sent = await self._send_audio(pcm, self._generation)
        if sent is None:
            return True  # interrupted by the caller
        await self._mark()
        self.stage("phrase", "ok", started, audio_bytes=sent)
        return True

    async def speak(self, text: str, *, cache: bool = False) -> float | None:
        """Speak ``text``; returns seconds until its first audio was sent (None
        if nothing could be spoken).  Pieces are synthesised one at a time and
        sent as soon as each is ready.  If synthesis fails before any audio
        went out, a cached apology is played instead (never silence)."""
        started = time.perf_counter()
        if cache:
            return (time.perf_counter() - started) if await self.phrase(text) else None
        generation = self._generation
        first_audio: float | None = None
        sent_total, error = 0, None
        for piece in speech_pieces(self.speech, text):
            try:
                pcm = await asyncio.wait_for(
                    asyncio.to_thread(self.speech.synthesize, speakable(piece), self.rate),
                    timeout=self._tts_deadline(),
                )
            except SpeechError as exc:
                error = getattr(exc, "reason", type(exc).__name__)
                break
            except TimeoutError:
                error = "timeout"
                break
            if not pcm:
                error = "empty_audio"
                break
            sent = await self._send_audio(pcm, generation)
            if sent is None:
                break  # barge-in: the rest of this reply is dropped
            sent_total += sent
            if first_audio is None:
                first_audio = time.perf_counter() - started
        if first_audio is not None:
            if generation == self._generation:
                await self._mark()
            self.stage("tts", "ok" if error is None else "partial", started, error=error,
                       audio_bytes=sent_total)  # fmt: skip
            return first_audio
        if error is None:
            return None  # interrupted before anything was sent: not a failure
        self.stage("tts", "failed", started, error=error, audio_bytes=0)
        await self.recover(SORRY_SPEECH, "tts_" + error)
        return None

    def _tts_deadline(self) -> float:
        """Most a synthesis request may take, even if it hangs below HTTP."""
        return self.config.speech_timeout_seconds + TTS_WATCHDOG_MARGIN

    async def recover(self, apology: str, reason: str) -> None:
        """A turn could not be answered: apologise, or end the call politely
        after repeated failures (or when no apology at all can play)."""
        self._failures += 1
        if self._failures >= MAX_CONSECUTIVE_FAILURES:
            await self._first_phrase(SORRY_UNAVAILABLE, SORRY_CLOSED)
            self.end(f"repeated_failures:{reason}")
        elif not await self._first_phrase(apology, *APOLOGIES):
            self.end(f"no_audio:{reason}")

    async def _first_phrase(self, *texts: str) -> bool:
        """Play the first of ``texts`` that has (or can get) audio."""
        for text in dict.fromkeys(texts):
            if await self.phrase(text):
                return True
        return False

    # ── Inbound events (called from the receive loop) ────────────────────────

    async def on_media(self, payload: str) -> None:
        try:
            pcm = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            self._bad_frame("invalid_base64")
            return
        if len(pcm) > MAX_MEDIA_BYTES:
            self._bad_frame("oversized")
            return
        for event in self.detector.feed(pcm):
            if event.kind == "speech_start" and self.playing:
                # Barge-in: stop what the bot is saying and anything queued.
                self.playing = False
                self._generation += 1
                await self.send({"event": "clear", "stream_sid": self.stream_sid})
                self.stage("barge_in", "ok")
            elif event.kind == "utterance":
                self._utterances += 1
                self.queue.put_nowait(("speech", (event.audio, time.perf_counter())))

    def _bad_frame(self, kind: str) -> None:
        self.bad_frames += 1
        if self.bad_frames == 1:  # log the first, count the rest
            self.stage("media", "ignored", error=kind)

    def on_mark(self, name: str | None) -> None:
        if name is not None and name == self._last_mark:
            self.playing = False

    def on_dtmf(self, digit: str | None) -> None:
        if digit:
            self.queue.put_nowait(("dtmf", digit))

    # ── Worker: one turn at a time, so turns never overlap ───────────────────

    async def run(self) -> None:
        try:
            if not await self._greet():
                return
            while not self.hangup:
                kind, value = await self.queue.get()
                self._turn_no += 1
                try:
                    if kind == "speech":
                        await self._handle_speech([value, *self._drain_speech()])
                    elif kind == "dtmf":
                        await self._handle_dtmf(value)
                except CallEnded:
                    raise
                except Exception:
                    # Any bug in one turn: apologise and keep the call usable.
                    logger.exception("voice call=%s turn=%d stage=turn result=error error=unexpected",
                                     self.call_sid, self._turn_no)  # fmt: skip
                    await self.recover(SORRY_RETRY, "unexpected")
        except CallEnded as exc:
            self.end(f"send_failed:{exc}")

    def _drain_speech(self) -> list[tuple[bytes, float]]:
        """Further utterances already queued (the caller kept talking): they
        are merged into the same turn instead of answered one by one."""
        items, others = [], []
        while not self.queue.empty():
            kind, value = self.queue.get_nowait()
            if kind == "speech":
                items.append(value)
            else:
                others.append((kind, value))
        for item in others:  # keypresses keep their turn after this one
            self.queue.put_nowait(item)
        return items

    async def _greet(self) -> bool:
        started = time.perf_counter()
        try:
            restaurant = await asyncio.to_thread(self._load_restaurant)
        except Exception:
            logger.exception("voice call=%s turn=0 stage=greeting result=error error=restaurant_lookup",
                             self.call_sid)  # fmt: skip
            restaurant = None
        if restaurant is None or not restaurant.is_active:
            await self.phrase(SORRY_CLOSED)
            self.end("restaurant_unavailable")
            return False
        if not await self.phrase(greeting_text(restaurant.name, self.config)):
            # Without a greeting the caller would hear only silence: end the
            # call (Exotel then moves on) rather than keep it open and mute.
            self.stage("greeting", "failed", started, error="no_audio")
            self.end("greeting_unavailable")
            return False
        self.stage("greeting", "ok", started)
        return True

    def _load_restaurant(self):
        with self.db_factory() as db:
            try:
                return get_restaurant(db, self.restaurant_id)
            except NotFoundError:
                return None

    async def _transcribe(self, items: list[tuple[bytes, float]]) -> str | None:
        """One STT request for one or more utterances (joined with a short
        pause).  None if nothing usable was heard (already recovered)."""
        audio = silence(200, self.rate).join(pcm for pcm, _ in items)
        started = time.perf_counter()
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(self.speech.transcribe, audio, self.rate),
                timeout=self.config.speech_timeout_seconds + STT_WATCHDOG_MARGIN,
            )
        except (SpeechError, TimeoutError) as exc:
            reason = "timeout" if isinstance(exc, TimeoutError) else getattr(exc, "reason", "error")
            self.stage("stt", "failed", started, error=reason)
            await self.recover(SORRY_NOT_HEARD, "stt_" + reason)
            return None
        if not text:
            self.stage("stt", "empty", started)
            spoken_ms = len(audio) * 1000 // (self.rate * SAMPLE_WIDTH)
            if spoken_ms >= EMPTY_TRANSCRIPT_REPLY_MS:  # real speech we could not read
                await self.recover(SORRY_NOT_HEARD, "stt_empty")
            return None
        self.stage("stt", "ok", started)
        return text

    async def _handle_speech(self, items: list[tuple[bytes, float]]) -> None:
        speech_end = items[-1][1]
        t = time.perf_counter()
        text = await self._transcribe(items)
        stt = time.perf_counter() - t
        if not text:
            return
        agent_s = 0.0
        while True:
            seen = self._utterances
            confirm = self.pending_proposal is not None and is_affirmative(text)
            t = time.perf_counter()
            outcome = await self._turn(text[:MAX_MESSAGE_CHARS], confirm=confirm)
            agent_s += time.perf_counter() - t
            if outcome is None:
                return
            reply, placed = outcome
            if self._utterances != seen and not placed and not self.queue.empty():
                # The caller said more while this reply was being prepared.
                # Answer the latest request instead of the stale one: forget
                # the unspoken reply and run one turn with everything said.
                newer = self._drain_speech()
                more = await self._transcribe(newer) if newer else None
                if more:
                    self._forget_last_turn(reply)
                    self.stage("stale_reply", "skipped")
                    text, speech_end = f"{text} {more}", newer[-1][1]
                    continue
            break
        first_audio = await self.speak(reply)
        self.turns += 1
        if first_audio is not None:
            self._failures = 0
        total = (time.perf_counter() - speech_end) if first_audio is not None else None
        logger.info(
            "voice call=%s turn=%d stage=turn result=%s end_of_speech_wait=%dms stt=%.2fs "
            "agent=%.2fs tts_first_audio=%s total_to_first_audio=%s reply_chars=%d",
            self.call_sid, self._turn_no, "ok" if first_audio is not None else "recovered",
            self.config.vad_silence_ms, stt, agent_s,
            f"{first_audio:.2f}s" if first_audio is not None else "none",
            f"{total:.2f}s" if total is not None else "none", len(reply),
        )  # fmt: skip

    async def _handle_dtmf(self, digit: str) -> None:
        if digit == "1" and self.pending_proposal is not None:
            outcome = await self._turn(None, confirm=True)
            if outcome is not None and await self.speak(outcome[0]) is not None:
                self._failures = 0

    async def _turn(self, text: str | None, *, confirm: bool) -> tuple[str, bool] | None:
        """Run one agent turn within ``VOICE_TURN_TIMEOUT_SECONDS``:
        (reply, order_placed), or None when the turn was recovered or the call
        is ending.  A late result from a timed-out turn is discarded."""
        started = time.perf_counter()
        task = asyncio.ensure_future(asyncio.to_thread(self._agent_turn, text, confirm))
        deadline = self.config.voice_turn_timeout_seconds
        # Short cached phrases while the reply is slow (e.g. the model is
        # waiting out a rate limit), so the caller never sits in silence.
        fillers = ((self.config.voice_filler_after_ms, FILLER),
                   (self.config.voice_still_working_after_ms, STILL_WORKING))  # fmt: skip
        for after_ms, filler in fillers:
            remaining = after_ms / 1000 - (time.perf_counter() - started)
            if after_ms <= 0 or after_ms / 1000 >= deadline or task.done():
                continue
            done, _ = await asyncio.wait({task}, timeout=max(0.0, remaining))
            if not done:
                await self.phrase(filler)
        try:
            remaining = max(0.0, deadline - (time.perf_counter() - started))
            reply, placed = await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except TimeoutError:
            self.stage("agent", "timeout", started, error="turn_deadline")
            await self.recover(SORRY_RETRY, "agent_timeout")
            return None
        except LLMError as exc:
            reason = str(getattr(exc, "reason", type(exc).__name__))
            self.stage("agent", "failed", started, error=reason)
            await self.recover(SORRY_RETRY, "llm_" + reason)
            return None
        except (ConflictError, NotFoundError) as exc:
            self.stage("agent", "refused", started, error=type(exc).__name__)
            return "Sorry, I couldn't do that. Could you tell me again what you'd like?", False
        self.stage("agent", "ok", started)
        return reply, placed

    def _forget_last_turn(self, reply: str) -> None:
        """Drop a reply the caller never heard from the conversation history."""
        session = self.store.get(self.restaurant_id, self.session_id) if self.session_id else None
        if session and session.turns and session.turns[-1][-1].get("content") == reply:
            session.turns.pop()

    def _agent_turn(self, text: str | None, confirm: bool) -> tuple[str, bool]:
        request = AgentChatRequest(
            message=text,
            session_id=self.session_id,
            confirm_proposal_id=self.pending_proposal if confirm else None,
        )
        with self.db_factory() as db:
            agent = RestaurantAgent(db, self.store, self.provider_factory)
            response = agent.chat(
                self.restaurant_id, request, caller_phone=self.caller, source=OrderSource.AI_VOICE
            )
        self.session_id = response.session_id
        previous = self.pending_proposal
        self.pending_proposal = (
            response.pending_order.proposal_id if response.requires_confirmation else None
        )
        placed = response.order is not None
        if placed:
            logger.info("voice call=%s stage=order result=ok order=%s", self.call_sid,
                        response.order.order_number)  # fmt: skip
        if self.pending_proposal is not None and self.pending_proposal != previous:
            return f"{response.reply} {CONFIRM_HINT}", placed
        return response.reply, placed


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
    connected_at = time.perf_counter()
    if not _authorized(websocket, config):
        logger.warning("voice stage=connect result=rejected error=bad_credentials")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    try:
        speech = speech_factory()
    except SpeechNotConfiguredError:
        logger.error("voice stage=connect result=rejected error=speech_not_configured")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    await websocket.accept()
    call: ExotelCall | None = None
    worker: asyncio.Task | None = None
    finished_at: float | None = None
    reason = "unknown"
    try:
        while True:
            # Bounded waits: Exotel must send "start" promptly, and a live
            # call streams media continuously, so a long gap means it is gone.
            timeout = config.voice_idle_timeout_seconds if call else config.voice_start_timeout_seconds
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
            except TimeoutError:
                reason = "idle_timeout" if call else "start_timeout"
                logger.warning("voice call=%s stage=connect result=timeout error=%s",
                               call.call_sid if call else "-", reason)  # fmt: skip
                break
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            event = message.get("event")

            if event == "start" and call is None:
                start = message.get("start") or {}
                restaurant_id = _restaurant_id(start.get("custom_parameters"), config)
                if restaurant_id is None:
                    logger.error("voice stage=connect result=rejected error=no_restaurant")
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
                logger.info("Exotel call %s started (caller %s) %.2fs after connect", call.call_sid,
                            mask_phone(call.caller), time.perf_counter() - connected_at)  # fmt: skip
                worker = asyncio.create_task(call.run())
            elif call is None:
                continue  # "connected", or anything before "start"
            elif event == "media":
                try:
                    await call.on_media((message.get("media") or {}).get("payload") or "")
                except CallEnded:
                    reason = "send_failed"
                    break
            elif event == "dtmf":
                call.on_dtmf((message.get("dtmf") or {}).get("digit"))
            elif event == "mark":
                call.on_mark((message.get("mark") or {}).get("name"))
            elif event == "stop":
                stop_reason = (message.get("stop") or {}).get("reason")
                reason = f"stopped:{stop_reason}"
                logger.info("Exotel call %s stopped (%s)", call.call_sid, stop_reason)
                break

            if worker is not None and worker.done():
                finished_at = finished_at or time.perf_counter()
                if call.playing and time.perf_counter() - finished_at < GOODBYE_GRACE_SECONDS:
                    continue  # let the last words finish (Exotel sends a mark)
                # The worker ended the conversation and its last words have
                # finished playing (or Exotel never confirmed them).
                if not worker.cancelled() and worker.exception() is not None:
                    logger.error("Exotel call %s worker crashed", call.call_sid,
                                 exc_info=worker.exception())  # fmt: skip
                reason = call.end_reason or "worker_finished"
                break
    except WebSocketDisconnect:
        reason = "disconnected"
    finally:
        if worker is not None and not worker.done():
            worker.cancel()
        if call is not None:
            logger.info(
                "voice call=%s stage=call result=ended reason=%s duration=%.1fs turns=%d "
                "audio_bytes=%d bad_frames=%d",
                call.call_sid, call.end_reason or reason, time.perf_counter() - call.started_at,
                call.turns, call.audio_bytes_sent, call.bad_frames,
            )  # fmt: skip
    try:
        await websocket.close()
    except RuntimeError:
        pass  # already closed
