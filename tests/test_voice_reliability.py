"""Voice-call reliability: a caller must never be left in silence.

Every failure path ends in a spoken recovery (cached audio) or a polite end of
the call: TTS failures and empty audio, STT timeouts and rate limits, slow or
failing assistant turns, unexpected errors, dead sockets, missing ``start``,
streams that stop, and malformed frames.  Fake providers and sockets; these
simulate the protocol and are not evidence of real telephone reliability.
"""

import asyncio
import base64
import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_voice_turns import FakeSpeech, FakeWS, make_call, scripted_turns, tone, utterance

from app.agent import service as agent_service
from app.agent.llm import LLMUnavailableError
from app.agent.sessions import InMemorySessionStore
from app.config import Settings
from app.main import app
from app.routers.agent import get_provider_factory, get_session_store
from app.telephony import exotel
from app.telephony.speech import SpeechUnavailableError

say = exotel.speakable


class SelectiveSpeech(FakeSpeech):
    """Fails (or returns nothing) only for the given texts; fixed phrases work."""

    def __init__(self, *args, fail=(), empty=(), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fail, self.empty = set(fail), set(empty)

    def synthesize(self, text: str, sample_rate: int) -> bytes:
        if text in self.fail:
            raise SpeechUnavailableError("rate_limited")
        if text in self.empty:
            return b""
        return super().synthesize(text, sample_rate)


# ── Speech synthesis failures ────────────────────────────────────────────────


class TestTtsFailures:
    async def test_failed_reply_gets_a_spoken_apology(self) -> None:
        speech = SelectiveSpeech(["menu please"], fail={say("We have dosas.")})
        ws = FakeWS()
        call = make_call(ws, speech)
        scripted_turns(call, [("We have dosas.", False)])
        await call._handle_speech([utterance()])
        assert speech.synthesized == [say(exotel.SORRY_SPEECH)]
        assert "media" in ws.events() and ws.events()[-1] == "mark"
        assert not call.hangup and call._failures == 1

    async def test_empty_audio_counts_as_a_failure(self, caplog) -> None:
        speech = SelectiveSpeech(["menu please"], empty={say("We have dosas.")})
        call = make_call(FakeWS(), speech)
        scripted_turns(call, [("We have dosas.", False)])
        with caplog.at_level("INFO", logger="app"):
            await call._handle_speech([utterance()])
        assert speech.synthesized == [say(exotel.SORRY_SPEECH)]
        assert "stage=tts result=failed" in caplog.text and "error=empty_audio" in caplog.text

    async def test_hung_synthesis_is_cut_off_with_an_apology(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(exotel, "TTS_WATCHDOG_MARGIN", 0.05)
        reply = say("We have dosas.")

        class Hanging(FakeSpeech):
            def synthesize(self, text, sample_rate):
                if text == reply:
                    time.sleep(0.5)
                return super().synthesize(text, sample_rate)

        speech = Hanging(["menu please"])
        ws = FakeWS()
        call = make_call(ws, speech, speech_timeout_seconds=0)
        scripted_turns(call, [("We have dosas.", False)])
        started = time.perf_counter()
        with caplog.at_level("INFO", logger="app"):
            await call._handle_speech([utterance()])
        assert time.perf_counter() - started < 0.45
        assert "stage=tts result=failed" in caplog.text and "error=timeout" in caplog.text
        assert say(exotel.SORRY_SPEECH) in speech.synthesized and "media" in ws.events()
        assert not call.hangup and call._failures == 1

    async def test_hung_phrase_synthesis_gives_up(self) -> None:
        class Hanging(FakeSpeech):
            def synthesize(self, text, sample_rate):
                time.sleep(0.5)
                return super().synthesize(text, sample_rate)

        started = time.perf_counter()
        assert await exotel.cached_phrase(Hanging(), exotel.FILLER, 8000, timeout=0.05) is None
        assert time.perf_counter() - started < 0.45

    async def test_repeated_failures_say_goodbye_and_end_the_call(self) -> None:
        reply = say("We have dosas.")
        speech = SelectiveSpeech(["menu", "menu"], fail={reply})
        call = make_call(FakeWS(), speech)
        scripted_turns(call, [("We have dosas.", False), ("We have dosas.", False)])
        await call._handle_speech([utterance()])
        await call._handle_speech([utterance()])
        assert speech.synthesized == [say(exotel.SORRY_SPEECH), say(exotel.SORRY_UNAVAILABLE)]
        assert call.hangup and call.end_reason == "repeated_failures:tts_rate_limited"

    async def test_success_resets_the_failure_count(self) -> None:
        bad, good = say("Bad."), say("Good.")
        speech = SelectiveSpeech(["a", "b", "c"], fail={bad})
        call = make_call(FakeWS(), speech)
        scripted_turns(call, [("Bad.", False), ("Good.", False), ("Bad.", False)])
        for _ in range(3):
            await call._handle_speech([utterance()])
        assert not call.hangup and call._failures == 1

    async def test_falls_back_to_any_cached_apology(self) -> None:
        reply = say("We have dosas.")
        speech = SelectiveSpeech(["menu"], fail={reply, say(exotel.SORRY_SPEECH)})
        ws = FakeWS()
        call = make_call(ws, speech)
        scripted_turns(call, [("We have dosas.", False)])
        await call._handle_speech([utterance()])
        assert speech.synthesized == [say(exotel.SORRY_RETRY)]  # next apology in line
        assert "media" in ws.events() and not call.hangup

    async def test_nothing_can_be_spoken_ends_the_call_instead_of_silence(self) -> None:
        call = make_call(FakeWS(), FakeSpeech(["hello"], fail_after=0))
        scripted_turns(call, [("Hi there.", False)])
        await call._handle_speech([utterance()])
        assert call.hangup and call.end_reason.startswith("no_audio:tts_")

    async def test_greeting_that_cannot_play_ends_the_call(self) -> None:
        call = make_call(FakeWS(), FakeSpeech(fail_after=0))
        call._load_restaurant = lambda: type("R", (), {"name": "Spice", "is_active": True})()
        assert await call._greet() is False
        assert call.hangup and call.end_reason == "greeting_unavailable"


# ── Phrase cache on disk ─────────────────────────────────────────────────────


class TestPhraseDiskCache:
    async def test_phrases_survive_a_restart_and_a_tts_outage(self, tmp_path) -> None:
        directory = str(tmp_path / "phrases")
        first = FakeSpeech()
        ready, failed = await exotel.prewarm_phrases(first, 8000, directory, ["Welcome to Spice."])
        assert (ready, failed) == (len(exotel.FIXED_PHRASES) + 1, 0)
        assert len(list((tmp_path / "phrases").glob("*.pcm"))) == ready

        restarted = FakeSpeech(fail_after=0)  # new process, TTS out of quota
        audio = await exotel.cached_phrase(restarted, exotel.SORRY_RETRY, 8000, directory)
        assert audio == b"\x01\x00" * 4000 and restarted.synthesized == []
        again, failed = await exotel.prewarm_phrases(restarted, 8000, directory, ["Welcome to Spice."])
        assert (again, failed) == (ready, 0)  # nothing re-synthesised

    async def test_corrupt_cache_file_is_ignored(self, tmp_path) -> None:
        speech = FakeSpeech()
        path = exotel._phrase_file(str(tmp_path), speech, exotel.FILLER, 8000)
        path.write_bytes(b"\x01")  # odd length: not 16-bit PCM
        audio = await exotel.cached_phrase(speech, exotel.FILLER, 8000, str(tmp_path))
        assert audio == b"\x01\x00" * 4000 and speech.synthesized == [say(exotel.FILLER)]
        assert path.read_bytes() == audio  # repaired

    async def test_call_uses_disk_phrases_when_tts_is_down(self, tmp_path) -> None:
        await exotel.prewarm_phrases(FakeSpeech(), 8000, str(tmp_path))
        speech = FakeSpeech(["menu"], fail_after=0)  # same voice, now out of quota
        ws = FakeWS()
        call = make_call(ws, speech, voice_phrase_cache_dir=str(tmp_path))
        scripted_turns(call, [("We have dosas.", False)])
        await call._handle_speech([utterance()])
        assert "media" in ws.events() and not call.hangup  # the apology played from disk


# ── Speech recognition ───────────────────────────────────────────────────────


class TestSttFailures:
    @pytest.mark.parametrize("reason", ["timeout", "rate_limited", "provider_error"])
    async def test_provider_failure_asks_to_repeat(self, reason) -> None:
        class Failing(FakeSpeech):
            def transcribe(self, pcm, rate):
                raise SpeechUnavailableError(reason)

        speech = Failing()
        call = make_call(FakeWS(), speech)
        await call._handle_speech([utterance()])
        assert speech.synthesized == [say(exotel.SORRY_NOT_HEARD)] and call._failures == 1

    async def test_hung_transcription_is_cut_off(self, monkeypatch) -> None:
        monkeypatch.setattr(exotel, "STT_WATCHDOG_MARGIN", 0.05)

        class Hanging(FakeSpeech):
            def transcribe(self, pcm, rate):
                time.sleep(0.5)
                return "too late"

        speech = Hanging()
        call = make_call(FakeWS(), speech, speech_timeout_seconds=0)
        started = time.perf_counter()
        await call._handle_speech([utterance()])
        assert time.perf_counter() - started < 0.45
        assert speech.synthesized == [say(exotel.SORRY_NOT_HEARD)]

    async def test_long_speech_transcribed_as_nothing_gets_a_reply(self) -> None:
        speech = FakeSpeech(["", ""])
        call = make_call(FakeWS(), speech)
        await call._handle_speech([(tone(200), time.perf_counter())])  # a short noise: ignored
        assert speech.synthesized == []
        await call._handle_speech([(tone(1500), time.perf_counter())])  # real speech we could not read
        assert speech.synthesized == [say(exotel.SORRY_NOT_HEARD)]


# ── Assistant turns ──────────────────────────────────────────────────────────


class TestAgentFailures:
    async def test_slow_turn_is_answered_with_an_apology(self, caplog) -> None:
        speech = FakeSpeech(["hello", "hello"])
        call = make_call(FakeWS(), speech, voice_turn_timeout_seconds=0.1)

        def slow():
            time.sleep(0.4)
            return ("Too late.", False)

        scripted_turns(call, [slow, ("On time.", False)])
        with caplog.at_level("INFO", logger="app"):
            started = time.perf_counter()
            await call._handle_speech([utterance()])
            assert time.perf_counter() - started < 0.35
        assert speech.synthesized == [say(exotel.SORRY_RETRY)]
        assert "stage=agent result=timeout" in caplog.text
        await asyncio.sleep(0.4)  # the late result is discarded, not spoken
        await call._handle_speech([utterance()])
        assert speech.synthesized == [say(exotel.SORRY_RETRY), "On time."] and call._failures == 0

    async def test_rate_limited_model_recovers(self) -> None:
        speech = FakeSpeech(["hi"])
        call = make_call(FakeWS(), speech)
        scripted_turns(call, [LLMUnavailableError("rate_limited")])
        await call._handle_speech([utterance()])
        assert speech.synthesized == [say(exotel.SORRY_RETRY)] and not call.hangup

    async def test_unexpected_error_keeps_the_call_alive(self, caplog) -> None:
        speech = FakeSpeech(["one", "two"])
        call = make_call(FakeWS(), speech)
        call._greet = lambda: asyncio.sleep(0, result=True)
        scripted_turns(call, [ValueError("bug"), ("Second turn works.", False)])
        worker = asyncio.create_task(call.run())
        call.queue.put_nowait(("speech", utterance()))
        await asyncio.sleep(0.1)
        call.queue.put_nowait(("speech", utterance()))
        await asyncio.sleep(0.1)
        worker.cancel()
        assert speech.synthesized == [say(exotel.SORRY_RETRY), "Second turn works."]
        assert "result=error error=unexpected" in caplog.text


# ── Sockets and frames ───────────────────────────────────────────────────────


class TestSocketAndFrames:
    async def test_dead_socket_ends_the_call(self) -> None:
        class DeadWS(FakeWS):
            async def send_text(self, text):
                raise RuntimeError("socket closed")

        call = make_call(DeadWS(), FakeSpeech(["hi"]))
        call._greet = lambda: asyncio.sleep(0, result=True)
        scripted_turns(call, [("Hello!", False)])
        call.queue.put_nowait(("speech", utterance()))
        await asyncio.wait_for(call.run(), timeout=2)
        assert call.hangup and call.end_reason == "send_failed:RuntimeError"

    async def test_bad_frames_are_ignored_and_counted(self, caplog) -> None:
        call = make_call(FakeWS(), FakeSpeech())
        with caplog.at_level("INFO", logger="app"):
            await call.on_media("not base64!!")
            await call.on_media(base64.b64encode(b"\x00" * (exotel.MAX_MEDIA_BYTES + 2)).decode())
            await call.on_media(base64.b64encode(b"\x00\x01\x02").decode())  # odd length: fine
        assert call.bad_frames == 2
        assert caplog.text.count("stage=media result=ignored") == 1  # logged once, counted after


# ── Endpoint: timeouts, concurrency, call summary ────────────────────────────


@pytest.fixture
def voice_app(monkeypatch):
    monkeypatch.setattr(agent_service, "load_menu_context", lambda db, restaurant: [])
    restaurant = type("R", (), {"name": "Spice", "is_active": True})()
    env = {"speech": FakeSpeech(), "settings": Settings(
        exotel_restaurant_id=uuid.uuid4(), exotel_ws_username=None, exotel_ws_password=None,
        voice_phrase_cache_dir=None, voice_start_timeout_seconds=0.3, voice_idle_timeout_seconds=0.5,
        vad_silence_ms=200, vad_min_speech_ms=100)}  # fmt: skip
    monkeypatch.setattr(exotel.ExotelCall, "_load_restaurant", lambda self: restaurant)
    app.dependency_overrides[exotel.get_speech_factory] = lambda: (lambda: env["speech"])
    app.dependency_overrides[exotel.get_settings] = lambda: env["settings"]
    app.dependency_overrides[exotel.get_db_factory] = lambda: None
    app.dependency_overrides[get_session_store] = lambda: InMemorySessionStore(ttl_seconds=60, max_sessions=10)
    app.dependency_overrides[get_provider_factory] = lambda: (lambda: None)
    try:
        yield env
    finally:
        for dep in (exotel.get_speech_factory, exotel.get_settings, exotel.get_db_factory,
                    get_session_store, get_provider_factory):  # fmt: skip
            app.dependency_overrides.pop(dep, None)


def _start(ws, call_sid="c1", caller="08000000001") -> None:
    ws.send_text(json.dumps({"event": "start", "stream_sid": "s", "start": {
        "stream_sid": "s", "call_sid": call_sid, "from": caller, "custom_parameters": {}}}))  # fmt: skip


def _until_mark(ws) -> list[dict]:
    out = []
    while True:
        out.append(ws.receive_json())
        if out[-1]["event"] == "mark":
            return out


class TestEndpoint:
    def test_socket_without_start_is_closed(self, voice_app, caplog) -> None:
        with caplog.at_level("INFO", logger="app"):
            with TestClient(app).websocket_connect(exotel.WS_PATH) as ws:
                ws.send_text(json.dumps({"event": "connected"}))
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_json()
        assert "error=start_timeout" in caplog.text

    def test_call_whose_stream_stops_is_closed_with_a_summary(self, voice_app, caplog) -> None:
        with caplog.at_level("INFO", logger="app"):
            with TestClient(app).websocket_connect(exotel.WS_PATH) as ws:
                _start(ws)
                greeting = _until_mark(ws)
                assert greeting[0]["event"] == "media"
                with pytest.raises(WebSocketDisconnect):  # no more media: idle timeout
                    ws.receive_json()
        summary = next(r.getMessage() for r in caplog.records if "stage=call result=ended" in r.getMessage())
        assert "reason=idle_timeout" in summary and "audio_bytes=" in summary
        assert "08000000001" not in caplog.text

    def test_greeting_failure_closes_the_call(self, voice_app) -> None:
        voice_app["speech"] = FakeSpeech(fail_after=0)
        with TestClient(app).websocket_connect(exotel.WS_PATH) as ws:
            _start(ws)
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()

    def test_two_calls_at_once_are_independent(self, voice_app) -> None:
        client = TestClient(app)
        with client.websocket_connect(exotel.WS_PATH) as a, client.websocket_connect(exotel.WS_PATH) as b:
            _start(a, "call-a")
            _start(b, "call-b")
            ga, gb = _until_mark(a), _until_mark(b)
            assert ga[-1]["mark"]["name"] == gb[-1]["mark"]["name"] == "reply-1"
            assert all(m["stream_sid"] == "s" for m in ga + gb)
