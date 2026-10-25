"""Exotel Voicebot integration: audio helpers, speech provider, and the call
WebSocket, with fake speech, a scripted or fake agent, and a mocked DB.

No PostgreSQL, no network, no API keys.
"""

import base64
import io
import json
import uuid
import wave
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from agent_fakes import ScriptedProvider, say
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketDisconnect

from app.agent import service as agent_service
from app.agent.prompts import caller_context
from app.agent.schemas import AgentChatResponse, PendingOrderSummary
from app.agent.sessions import InMemorySessionStore
from app.config import Settings
from app.main import app
from app.models import OrderSource
from app.routers.agent import get_provider_factory, get_session_store
from app.telephony import exotel
from app.telephony.audio import (
    UtteranceDetector,
    exotel_chunks,
    from_wav,
    resample,
    rms,
    to_wav,
)
from app.telephony.speech import (
    OpenAICompatibleSpeech,
    SpeechNotConfiguredError,
    SpeechUnavailableError,
    speakable,
    split_text,
)

RID = uuid.uuid4()
RESTAURANT = SimpleNamespace(id=RID, name="Spice Garden", currency="INR", is_active=True)
RATE = 8000


def tone(ms: int, amplitude: int = 3000) -> bytes:
    return array("h", [amplitude, -amplitude] * (RATE * ms // 2000)).tobytes()


def quiet(ms: int) -> bytes:
    return bytes(RATE * ms // 1000 * 2)


# ── Audio helpers ────────────────────────────────────────────────────────────


class TestAudio:
    def test_rms(self) -> None:
        assert rms(b"") == 0
        assert rms(quiet(20)) == 0
        assert rms(tone(20, 1000)) == pytest.approx(1000)

    def test_resample_changes_length_by_rate_ratio(self) -> None:
        pcm = tone(300)
        assert len(resample(pcm, 24000, 8000)) == len(pcm) // 3
        assert resample(pcm, 8000, 8000) == pcm
        assert resample(b"", 24000, 8000) == b""

    def test_to_wav(self) -> None:
        with wave.open(io.BytesIO(to_wav(tone(100), RATE))) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, RATE)
            assert wav.getnframes() == 800

    def test_from_wav_round_trip_and_streaming_header(self) -> None:
        data = to_wav(tone(100), 16000)
        assert from_wav(data) == (tone(100), 16000)
        # Streaming TTS writes 0xFFFFFFFF as the data size; still readable.
        streaming = data[:40] + b"\xff\xff\xff\xff" + data[44:]
        assert from_wav(streaming) == (tone(100), 16000)
        with pytest.raises(ValueError):
            from_wav(b"not a wav")

    @pytest.mark.parametrize("length", [1, 319, 3200, 8000, 8001, 25000])
    def test_exotel_chunks_meet_size_rules(self, length) -> None:
        chunks = exotel_chunks(b"\x01" * length)
        assert all(len(c) % 320 == 0 and 3200 <= len(c) <= 100_000 for c in chunks)
        assert b"".join(chunks)[:length] == b"\x01" * length
        assert exotel_chunks(b"") == []

    def test_detector_finds_one_utterance(self) -> None:
        vad = UtteranceDetector(sample_rate=RATE, energy_threshold=450, silence_ms=200,
                                min_speech_ms=100, max_utterance_ms=5000)  # fmt: skip
        events = vad.feed(quiet(100)) + vad.feed(tone(300)) + vad.feed(quiet(300))
        assert [e.kind for e in events] == ["speech_start", "utterance"]
        assert len(events[1].audio) >= len(tone(300))

    def test_detector_ignores_short_noise(self) -> None:
        vad = UtteranceDetector(sample_rate=RATE, energy_threshold=450, silence_ms=200,
                                min_speech_ms=100, max_utterance_ms=5000)  # fmt: skip
        assert vad.feed(tone(40) + quiet(400)) == []

    def test_detector_caps_long_speech(self) -> None:
        vad = UtteranceDetector(sample_rate=RATE, energy_threshold=450, silence_ms=200,
                                min_speech_ms=20, max_utterance_ms=500)  # fmt: skip
        kinds = [e.kind for e in vad.feed(tone(1200))]
        assert kinds.count("utterance") == 2


# ── Voice confirmation ───────────────────────────────────────────────────────


class TestAffirmative:
    @pytest.mark.parametrize("text", ["Yes", "yes please.", "Yeah, place the order", "Okay",
                                      "haan ji", "Sure, go ahead", "That's right"])  # fmt: skip
    def test_yes(self, text) -> None:
        assert exotel.is_affirmative(text)

    @pytest.mark.parametrize("text", ["", "no", "yes but add a lassi", "not yet",
                                      "please", "what is the total?", "the order",
                                      "yes yes yes yes yes yes yes"])  # fmt: skip
    def test_not_yes(self, text) -> None:
        assert not exotel.is_affirmative(text)


# ── Speech provider ──────────────────────────────────────────────────────────


def speech_settings(**overrides) -> Settings:
    values = dict(speech_provider="openai", speech_api_key="test-key", stt_model="stt",
                  tts_model="tts", tts_sample_rate=24000)  # fmt: skip
    return Settings(**(values | overrides))


class TestSpeechProvider:
    def test_requires_configuration(self) -> None:
        with pytest.raises(SpeechNotConfiguredError):
            OpenAICompatibleSpeech(speech_settings(speech_api_key=None))
        with pytest.raises(SpeechNotConfiguredError):
            OpenAICompatibleSpeech(speech_settings(tts_model=None))

    def test_transcribe_and_synthesize(self) -> None:
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.path.endswith("/audio/transcriptions"):
                return httpx.Response(200, json={"text": "  two dosas  "})
            return httpx.Response(200, content=tone(300))  # pretend 24 kHz pcm

        client = httpx.Client(transport=httpx.MockTransport(handler))
        speech = OpenAICompatibleSpeech(speech_settings(), client=client)
        assert speech.transcribe(tone(200), RATE) == "two dosas"
        assert len(speech.synthesize("hello", RATE)) == len(tone(300)) // 3
        assert all(r.headers["authorization"] == "Bearer test-key" for r in seen)
        assert b'"response_format":"pcm"' in seen[1].content.replace(b" ", b"")

    def test_groq_requests_wav_in_200_char_pieces_and_resamples(self) -> None:
        bodies = []
        rendered = tone(300)  # what Orpheus returns: 24 kHz, truthful header

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, content=to_wav(rendered, 24000))

        client = httpx.Client(transport=httpx.MockTransport(handler))
        speech = OpenAICompatibleSpeech(speech_settings(speech_provider="groq"), client=client)
        text = "Two masala dosas and one mango lassi. " * 8
        pcm = speech.synthesize(text, RATE)
        assert len(bodies) > 1 and all(len(b["input"]) <= 200 for b in bodies)
        assert " ".join(b["input"] for b in bodies) == text.strip()
        # sample_rate would only relabel Groq's header, so it is never sent.
        assert all(b["response_format"] == "wav" and "sample_rate" not in b for b in bodies)
        assert pcm == resample(rendered, 24000, RATE) * len(bodies)

    def test_groq_rejects_non_wav_audio(self) -> None:
        client = httpx.Client(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=b"ID3 not a wav")))  # fmt: skip
        speech = OpenAICompatibleSpeech(speech_settings(speech_provider="groq"), client=client)
        with pytest.raises(SpeechUnavailableError) as exc:
            speech.synthesize("hi", RATE)
        assert exc.value.reason == "invalid_audio"

    @pytest.mark.parametrize(("text", "spoken"), [
        ("Your order **ORD‑101168** is **confirmed**.", "Your order order 1 0 1 1 6 8 is confirmed."),
        ("Total: INR 235.00", "Total: 235 rupees."),
        ("It costs INR 1,241.50 today.", "It costs 1241 rupees 50 paise today."),
        ("Here it is:\n\n- 2 × Masala Dosa – INR 80.00\n- 1 × Tea\n\nAnything else?",
         "Here it is: 2 Masala Dosa, 80 rupees. 1 Tea. Anything else?"),
        ("## Menu\n* Idli", "Menu. Idli."),
        ("Tea (\u20b920) and Idli (\u20b940).", "Tea (20 rupees) and Idli (40 rupees)."),
        ("Only Rs. 1,250.50 today", "Only 1250 rupees 50 paise today."),
    ])  # fmt: skip
    def test_speakable(self, text, spoken) -> None:
        assert speakable(text) == spoken

    @pytest.mark.parametrize("text", ["", "Hello.", "word " * 120, "x" * 450,
                                      "Short one. " + "y" * 250 + ". End."])  # fmt: skip
    def test_split_text(self, text) -> None:
        pieces = split_text(text, 200)
        assert all(0 < len(p) <= 200 for p in pieces)
        assert "".join(pieces).replace(" ", "") == text.replace(" ", "")

    @pytest.mark.parametrize("status, reason", [(401, "authentication_failed"),
                                                (429, "rate_limited"), (500, "provider_error")])  # fmt: skip
    def test_errors_are_safe(self, status, reason) -> None:
        client = httpx.Client(transport=httpx.MockTransport(
            lambda r: httpx.Response(status, text="secret provider detail")))  # fmt: skip
        speech = OpenAICompatibleSpeech(speech_settings(), client=client)
        with pytest.raises(SpeechUnavailableError) as exc:
            speech.synthesize("hi", RATE)
        assert exc.value.reason == reason
        assert "secret" not in str(exc.value)

    def test_short_rate_limit_is_waited_out_once(self) -> None:
        queue = [httpx.Response(429, headers={"retry-after": "2"}), httpx.Response(200, content=tone(300))]
        waits: list = []
        client = httpx.Client(transport=httpx.MockTransport(lambda r: queue.pop(0)))
        speech = OpenAICompatibleSpeech(speech_settings(), client=client, sleep=waits.append)
        assert speech.synthesize("hello", RATE) and waits == [2.0]

    @pytest.mark.parametrize("responses", [
        [httpx.Response(429, headers={"retry-after": "30"})],  # too long: fail fast
        [httpx.Response(429)],  # no Retry-After
        [httpx.Response(429, headers={"retry-after": "1"})] * 2,  # retried only once
    ])  # fmt: skip
    def test_other_rate_limits_fail(self, responses) -> None:
        queue = list(responses)
        waits: list = []
        client = httpx.Client(transport=httpx.MockTransport(lambda r: queue.pop(0)))
        speech = OpenAICompatibleSpeech(speech_settings(), client=client, sleep=waits.append)
        with pytest.raises(SpeechUnavailableError) as exc:
            speech.synthesize("hi", RATE)
        assert exc.value.reason == "rate_limited" and len(waits) == len(responses) - 1


# ── WebSocket call ───────────────────────────────────────────────────────────


class FakeSpeech:
    def __init__(self, *transcripts: str) -> None:
        self.transcripts = list(transcripts)
        self.spoken: list[str] = []

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        return self.transcripts.pop(0)

    def synthesize(self, text: str, sample_rate: int) -> bytes:
        self.spoken.append(text)
        return b"\x01\x00" * 1600


class FakeAgent:
    """Stands in for RestaurantAgent: records requests, replays responses."""

    requests: list = []
    responses: list = []
    sources: list = []

    def __init__(self, db, store, provider_factory) -> None:
        pass

    def chat(self, restaurant_id, request, *, caller_phone=None, source=None):
        FakeAgent.sources.append(source)
        FakeAgent.requests.append((request, caller_phone))
        return FakeAgent.responses.pop(0)


def proposal(proposal_id: str = "a" * 32) -> PendingOrderSummary:
    return PendingOrderSummary(
        proposal_id=proposal_id, items=[], subtotal_paise=24100, total_paise=24100,
        total="INR 241.00", currency="INR", customer_name=None, customer_phone="09876543210",
        notes=None,
    )  # fmt: skip


def reply(text: str, pending: PendingOrderSummary | None = None) -> AgentChatResponse:
    return AgentChatResponse(reply=text, session_id="s" * 32,
                             requires_confirmation=pending is not None, pending_order=pending)  # fmt: skip


@pytest.fixture
def call_env(monkeypatch):
    """Install fakes; returns a dict to configure speech/provider/settings."""
    monkeypatch.setattr(agent_service, "load_menu_context", lambda db, restaurant: [])
    db = MagicMock(spec=Session)
    db.get.return_value = RESTAURANT
    env = {
        "speech": FakeSpeech(),
        "provider": ScriptedProvider(),
        "settings": Settings(exotel_restaurant_id=RID, exotel_ws_username=None,
                             exotel_ws_password=None, vad_silence_ms=200,
                             vad_min_speech_ms=100, exotel_greeting=None,
                             voice_phrase_cache_dir=None),  # fmt: skip
        "db": db,
    }

    @contextmanager
    def db_factory():
        yield db

    def speech_factory():
        if isinstance(env["speech"], Exception):
            raise env["speech"]
        return env["speech"]

    store = InMemorySessionStore(ttl_seconds=60, max_sessions=10)
    app.dependency_overrides[exotel.get_db_factory] = lambda: db_factory
    app.dependency_overrides[exotel.get_speech_factory] = lambda: speech_factory
    app.dependency_overrides[exotel.get_settings] = lambda: env["settings"]
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_provider_factory] = lambda: (lambda: env["provider"])
    FakeAgent.requests, FakeAgent.responses, FakeAgent.sources = [], [], []
    try:
        yield env
    finally:
        app.dependency_overrides.clear()


def start(ws, caller="09876543210", custom=None) -> None:
    ws.send_text(json.dumps({"event": "connected"}))
    ws.send_text(json.dumps({"event": "start", "sequence_number": 1, "stream_sid": "st1",
                             "start": {"stream_sid": "st1", "call_sid": "c1", "account_sid": "a",
                                       "from": caller, "to": "08000000000",
                                       "custom_parameters": custom or {},
                                       "media_format": {"encoding": "raw", "sample_rate": "8000",
                                                        "bit_rate": "128"}}}))  # fmt: skip


def speak(ws, ms: int = 300) -> None:
    for chunk in (tone(ms), quiet(400)):
        ws.send_text(json.dumps({"event": "media", "stream_sid": "st1",
                                 "media": {"chunk": 1, "timestamp": "0",
                                           "payload": base64.b64encode(chunk).decode()}}))  # fmt: skip


def receive_reply(ws) -> list[dict]:
    """Messages up to and including the next mark."""
    messages = []
    while True:
        message = ws.receive_json()
        messages.append(message)
        if message["event"] == "mark":
            return messages


def finish_playback(ws, mark: dict) -> None:
    ws.send_text(json.dumps({"event": "mark", "stream_sid": "st1", "mark": mark["mark"]}))


class TestExotelCall:
    def test_greets_then_answers_with_real_agent(self, call_env) -> None:
        call_env["speech"] = FakeSpeech("Do you have dosa?")
        call_env["provider"] = ScriptedProvider(say("Yes, we have masala dosa."))
        with TestClient(app).websocket_connect(exotel.WS_PATH) as ws:
            start(ws)
            greeting = receive_reply(ws)
            assert greeting[0] == {"event": "media", "stream_sid": "st1",
                                   "media": {"payload": greeting[0]["media"]["payload"]}}  # fmt: skip
            assert all(len(base64.b64decode(m["media"]["payload"])) % 320 == 0
                       for m in greeting[:-1])  # fmt: skip
            finish_playback(ws, greeting[-1])
            speak(ws)
            receive_reply(ws)
        assert call_env["speech"].spoken[0].startswith("Welcome to Spice Garden.")
        assert call_env["speech"].spoken[1] == "Yes, we have masala dosa."
        messages = call_env["provider"].requests[0]["messages"]
        assert messages[2] == {"role": "system", "content": caller_context("09876543210")}
        assert messages[-1] == {"role": "user", "content": "Do you have dosa?"}

    def test_voice_yes_confirms_pending_order(self, call_env, monkeypatch) -> None:
        monkeypatch.setattr(exotel, "RestaurantAgent", FakeAgent)
        call_env["speech"] = FakeSpeech("Two masala dosas", "Yes please")
        FakeAgent.responses = [reply("Two masala dosas, INR 241.00. Shall I place it?", proposal()),
                               reply("Thank you, your order has been placed.")]  # fmt: skip
        with TestClient(app).websocket_connect(exotel.WS_PATH) as ws:
            start(ws)
            finish_playback(ws, receive_reply(ws)[-1])
            speak(ws)
            finish_playback(ws, receive_reply(ws)[-1])
            speak(ws)
            receive_reply(ws)
        (first, phone), (second, _) = FakeAgent.requests
        assert phone == "09876543210"
        assert first.confirm_proposal_id is None and first.session_id is None
        assert second.confirm_proposal_id == "a" * 32 and second.session_id == "s" * 32
        assert set(FakeAgent.sources) == {OrderSource.AI_VOICE}  # phone orders are labelled
        assert call_env["speech"].spoken[1].endswith(exotel.CONFIRM_HINT)

    def test_dtmf_1_confirms_and_other_speech_does_not(self, call_env, monkeypatch) -> None:
        monkeypatch.setattr(exotel, "RestaurantAgent", FakeAgent)
        call_env["speech"] = FakeSpeech("One lassi", "yes but add a samosa")
        FakeAgent.responses = [reply("One lassi?", proposal("b" * 32)),
                               reply("Updated.", proposal("c" * 32)),
                               reply("Placed.")]  # fmt: skip
        with TestClient(app).websocket_connect(exotel.WS_PATH) as ws:
            start(ws)
            finish_playback(ws, receive_reply(ws)[-1])
            speak(ws)
            finish_playback(ws, receive_reply(ws)[-1])
            speak(ws)
            finish_playback(ws, receive_reply(ws)[-1])
            ws.send_text(json.dumps({"event": "dtmf", "stream_sid": "st1",
                                     "dtmf": {"digit": "1", "duration": "200"}}))  # fmt: skip
            receive_reply(ws)
        confirms = [r.confirm_proposal_id for r, _ in FakeAgent.requests]
        assert confirms == [None, None, "c" * 32]

    def test_barge_in_clears_playback(self, call_env, monkeypatch) -> None:
        monkeypatch.setattr(exotel, "RestaurantAgent", FakeAgent)
        call_env["speech"] = FakeSpeech("hello")
        FakeAgent.responses = [reply("Hi!")]
        with TestClient(app).websocket_connect(exotel.WS_PATH) as ws:
            start(ws)
            receive_reply(ws)  # greeting still "playing": no mark sent back
            speak(ws)
            assert ws.receive_json() == {"event": "clear", "stream_sid": "st1"}
            receive_reply(ws)

    def test_unknown_restaurant_says_closed_and_hangs_up(self, call_env) -> None:
        call_env["db"].get.return_value = None
        with TestClient(app).websocket_connect(exotel.WS_PATH) as ws:
            start(ws)
            finish_playback(ws, receive_reply(ws)[-1])
            ws.send_text(json.dumps({"event": "media", "stream_sid": "st1",
                                     "media": {"payload": base64.b64encode(quiet(20)).decode()}}))  # fmt: skip
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
        assert call_env["speech"].spoken == [exotel.SORRY_CLOSED]

    def test_restaurant_id_custom_parameter(self, call_env) -> None:
        call_env["settings"].exotel_restaurant_id = None
        with TestClient(app).websocket_connect(exotel.WS_PATH) as ws:
            start(ws, custom={"restaurant_id": str(RID)})
            receive_reply(ws)
        call_env["db"].get.assert_called()

    def test_no_restaurant_configured_rejects(self, call_env) -> None:
        call_env["settings"].exotel_restaurant_id = None
        with TestClient(app).websocket_connect(exotel.WS_PATH) as ws:
            start(ws)
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
        assert exc.value.code == 1008

    def test_speech_not_configured_rejects(self, call_env) -> None:
        call_env["speech"] = SpeechNotConfiguredError()
        with pytest.raises(WebSocketDisconnect):
            with TestClient(app).websocket_connect(exotel.WS_PATH):
                pass


class TestCallerId:
    """Any caller Exotel connects is served; caller ID only pre-fills the
    contact number and is never treated as identity."""

    @pytest.mark.parametrize("caller", ["08000000001", "09000000201", "+91 90000 00201"])
    def test_any_caller_is_served_and_number_passed_through(self, call_env, monkeypatch, caller) -> None:
        monkeypatch.setattr(exotel, "RestaurantAgent", FakeAgent)
        call_env["speech"] = FakeSpeech("One tea")
        FakeAgent.responses = [reply("One tea?")]
        with TestClient(app).websocket_connect(exotel.WS_PATH) as ws:
            start(ws, caller=caller)
            finish_playback(ws, receive_reply(ws)[-1])
            speak(ws)
            receive_reply(ws)
        assert FakeAgent.requests[0][1] == caller

    @pytest.mark.parametrize("raw", [None, "", "anonymous", "unknown", "+91", "12",
                                     "0900<script>", 9000000201, "1" * 25])  # fmt: skip
    def test_withheld_or_malformed_caller_id_still_served(self, call_env, monkeypatch, raw) -> None:
        monkeypatch.setattr(exotel, "RestaurantAgent", FakeAgent)
        call_env["speech"] = FakeSpeech("One tea")
        FakeAgent.responses = [reply("Sure. What number should I use for the order?")]
        with TestClient(app).websocket_connect(exotel.WS_PATH) as ws:
            start(ws, caller=raw)
            finish_playback(ws, receive_reply(ws)[-1])
            speak(ws)
            receive_reply(ws)
        assert FakeAgent.requests[0][1] is None  # the agent asks for a number instead

    def test_logs_mask_the_caller_number(self, call_env, caplog) -> None:
        with caplog.at_level("INFO", logger="app"):
            with TestClient(app).websocket_connect(exotel.WS_PATH) as ws:
                start(ws, caller="09000000201")
                receive_reply(ws)
        assert "caller ***0201" in caplog.text
        assert "9000000201" not in caplog.text

    def test_mask_phone(self) -> None:
        assert exotel.mask_phone("+91 90000 00201") == "***0201"
        assert exotel.mask_phone(None) == exotel.mask_phone("12") == "unknown"

    def test_prompt_never_presents_caller_id_as_verification(self) -> None:
        note = caller_context("09000000201")
        assert "not proof of identity" in note and "never tell the customer they are verified" in note
        assert "verified caller" not in note.lower()


class TestAuth:
    @pytest.fixture
    def secured(self, call_env):
        call_env["settings"].exotel_ws_username = "exotel"
        call_env["settings"].exotel_ws_password = SecretStr("s3cret")
        return call_env

    @pytest.mark.parametrize("header", [None, "Bearer x", "Basic !!!",
                                        "Basic " + base64.b64encode(b"exotel:wrong").decode()])  # fmt: skip
    def test_rejects_bad_credentials(self, secured, header) -> None:
        headers = {"Authorization": header} if header else {}
        with pytest.raises(WebSocketDisconnect) as exc:
            with TestClient(app).websocket_connect(exotel.WS_PATH, headers=headers):
                pass
        assert exc.value.code == 1008

    def test_accepts_basic_auth(self, secured) -> None:
        token = base64.b64encode(b"exotel:s3cret").decode()
        with TestClient(app).websocket_connect(
            exotel.WS_PATH, headers={"Authorization": f"Basic {token}"}
        ) as ws:
            start(ws)
            assert receive_reply(ws)[-1]["event"] == "mark"
