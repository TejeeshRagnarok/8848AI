"""Turn handling inside one call: streaming speech, barge-in cancellation,
stale replies, merged utterances, cached phrases, recovery and timings.

The call object is driven directly with a fake WebSocket, fake speech and a
fake agent turn, so the order of events is deterministic.  These simulate
the protocol; they are not evidence of real telephone barge-in behaviour.
"""

import asyncio
import base64
import json
import time
import uuid
from array import array

import pytest

from app.agent.llm import LLMResponseError, LLMUnavailableError
from app.config import Settings
from app.telephony import exotel
from app.telephony.audio import UtteranceDetector
from app.telephony.speech import SpeechUnavailableError

RATE = 8000


def tone(ms: int, amplitude: int = 3000) -> bytes:
    return array("h", [amplitude, -amplitude] * (RATE * ms // 2000)).tobytes()


def quiet(ms: int) -> bytes:
    return bytes(RATE * ms // 1000 * 2)


class FakeWS:
    def __init__(self, on_media=None) -> None:
        self.sent: list[dict] = []
        self.on_media = on_media

    async def send_text(self, text: str) -> None:
        message = json.loads(text)
        self.sent.append(message)
        if message["event"] == "media" and self.on_media:
            self.on_media(message)

    def events(self) -> list[str]:
        return [m["event"] for m in self.sent]


class FakeSpeech:
    """Each synthesised piece is 0.5 s of audio; records what it was asked."""

    def __init__(self, transcripts=(), split=None, fail_after=None) -> None:
        self.transcripts = list(transcripts)
        self.transcribed: list[int] = []  # audio lengths sent to STT
        self.synthesized: list[str] = []
        self.split = split
        self.fail_after = fail_after
        self.log: list[str] = []

    def pieces(self, text: str) -> list[str]:
        return self.split(text) if self.split else [text]

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        self.transcribed.append(len(pcm))
        return self.transcripts.pop(0) if self.transcripts else ""

    def synthesize(self, text: str, sample_rate: int) -> bytes:
        if self.fail_after is not None and len(self.synthesized) >= self.fail_after:
            raise SpeechUnavailableError("rate_limited")
        self.synthesized.append(text)
        self.log.append(f"synth:{text}")
        return b"\x01\x00" * 4000  # 0.5 s


def make_call(ws, speech, **settings) -> exotel.ExotelCall:
    config = Settings(exotel_restaurant_id=uuid.uuid4(), voice_filler_after_ms=0,
                      vad_silence_ms=200, vad_min_speech_ms=100,
                      voice_phrase_cache_dir=None).model_copy(update=settings)  # fmt: skip
    store = exotel.SessionStore  # unused by the fake agent turns below
    return exotel.ExotelCall(ws, stream_sid="s1", call_sid="c1", restaurant_id=config.exotel_restaurant_id,
                             caller="08000000001", speech=speech, store=store, provider_factory=None,
                             db_factory=None, config=config)  # fmt: skip


def utterance(ms: int = 300) -> tuple[bytes, float]:
    return tone(ms), time.perf_counter()


# ── Speaking ─────────────────────────────────────────────────────────────────


class TestSpeaking:
    async def test_pieces_are_sent_as_soon_as_each_is_ready(self) -> None:
        speech = FakeSpeech(split=lambda t: t.split("|"))
        ws = FakeWS(on_media=lambda m: speech.log.append("media"))
        call = make_call(ws, speech)
        first = await call.speak("First part.|Second part.")
        assert first is not None
        # The first piece's audio goes out before the second is synthesised.
        assert speech.log.index("media") < speech.log.index("synth:Second part.")
        assert ws.events()[-1] == "mark" and ws.events().count("mark") == 1

    async def test_chunks_are_small_for_quick_barge_in(self) -> None:
        ws = FakeWS()
        await make_call(ws, FakeSpeech()).speak("Hello there.")
        sizes = [len(base64.b64decode(m["media"]["payload"])) for m in ws.sent if m["event"] == "media"]
        assert sizes and all(s == 3200 for s in sizes)  # 200 ms each

    async def test_barge_in_stops_the_rest_of_the_reply(self) -> None:
        speech = FakeSpeech(split=lambda t: t.split("|"))
        ws = FakeWS()
        call = make_call(ws, speech)
        ws.on_media = lambda m: setattr(call, "_generation", call._generation + 1)  # caller cuts in
        await call.speak("One.|Two.|Three.")
        assert speech.synthesized == ["One."]  # nothing more generated or sent
        assert "mark" not in ws.events()

    async def test_caller_speech_while_playing_sends_clear(self) -> None:
        ws = FakeWS()
        call = make_call(ws, FakeSpeech())
        await call.speak("A long answer.")
        assert call.playing
        await call.on_media(base64.b64encode(tone(300)).decode())
        assert ws.events()[-1] == "clear" and not call.playing and call._generation == 1


class TestPhraseCache:
    async def test_fixed_phrases_are_synthesised_once_and_survive_tts_outage(self) -> None:
        speech = FakeSpeech()
        await make_call(FakeWS(), speech).speak(exotel.SORRY_RETRY, cache=True)
        speech.fail_after = 0  # TTS quota exhausted from now on
        ws = FakeWS()
        first = await make_call(ws, speech).speak(exotel.SORRY_RETRY, cache=True)
        assert first is not None and "media" in ws.events()
        assert speech.synthesized.count(exotel.speakable(exotel.SORRY_RETRY)) == 1

    async def test_uncached_reply_with_tts_outage_sends_nothing(self) -> None:
        ws = FakeWS()
        assert await make_call(ws, FakeSpeech(fail_after=0)).speak("Anything.") is None
        assert ws.sent == []


# ── Turns ────────────────────────────────────────────────────────────────────


def scripted_turns(call, script):
    """Replace the agent with a script of (reply, placed) or exceptions;
    a callable step runs first (to simulate the caller speaking meanwhile)."""
    calls: list[tuple] = []

    def fake(text, confirm):
        calls.append((text, confirm))
        step = script.pop(0)
        if callable(step):
            step = step()
        if isinstance(step, Exception):
            raise step
        return step

    call._agent_turn = fake
    return calls


class TestTurns:
    async def test_utterances_queued_together_become_one_turn(self) -> None:
        speech = FakeSpeech(["Two dosas and a coffee"])
        ws = FakeWS()
        call = make_call(ws, speech)
        calls = scripted_turns(call, [("Two masala dosas and a coffee, INR 185.00.", False)])
        a, b = utterance(), utterance()
        call.queue.put_nowait(("speech", b))
        await call._handle_speech([a, *call._drain_speech()])
        assert len(speech.transcribed) == 1  # one STT request for both
        assert speech.transcribed[0] > len(a[0]) + len(b[0])  # joined with a pause
        assert calls == [("Two dosas and a coffee", False)]

    async def test_stale_reply_is_dropped_for_the_latest_request(self) -> None:
        speech = FakeSpeech(["Two dosas", "actually make it three"])
        ws = FakeWS()
        call = make_call(ws, speech)

        def caller_keeps_talking():
            call._utterances += 1
            call.queue.put_nowait(("speech", utterance()))
            return ("Two masala dosas, INR 160.00.", False)

        calls = scripted_turns(call, [caller_keeps_talking, ("Three masala dosas, INR 240.00.", False)])
        forgotten = []
        call._forget_last_turn = forgotten.append
        await call._handle_speech([utterance()])
        assert [c[0] for c in calls] == ["Two dosas", "Two dosas actually make it three"]
        assert speech.synthesized == [exotel.speakable("Three masala dosas, INR 240.00.")]  # stale never spoken
        assert forgotten == ["Two masala dosas, INR 160.00."]

    async def test_noise_does_not_cancel_a_reply(self) -> None:
        speech = FakeSpeech(["What do you recommend?", ""])  # second "utterance" was noise
        call = make_call(FakeWS(), speech)

        def noise_meanwhile():
            call._utterances += 1
            call.queue.put_nowait(("speech", utterance()))
            return ("Try the masala dosa.", False)

        scripted_turns(call, [noise_meanwhile])
        await call._handle_speech([utterance()])
        assert speech.synthesized == ["Try the masala dosa."]

    async def test_order_placement_reply_is_never_dropped(self) -> None:
        speech = FakeSpeech(["yes", "and a tea"])
        call = make_call(FakeWS(), speech)
        call.pending_proposal = "p" * 32

        def placed_while_caller_talks():
            call._utterances += 1
            call.queue.put_nowait(("speech", utterance()))
            return ("Your order ORD-1 has been placed.", True)

        calls = scripted_turns(call, [placed_while_caller_talks])
        await call._handle_speech([utterance()])
        assert calls == [("yes", True)]
        assert speech.synthesized == [exotel.speakable("Your order ORD-1 has been placed.")]
        assert not call.queue.empty()  # "and a tea" is handled as the next turn

    async def test_assistant_outage_apologises_then_recovers(self) -> None:
        speech = FakeSpeech(["menu please", "menu please"])
        call = make_call(FakeWS(), speech)
        scripted_turns(call, [LLMUnavailableError("rate_limited"), ("We have dosas and idli.", False)])
        await call._handle_speech([utterance()])
        assert not call.hangup and speech.synthesized == [exotel.speakable(exotel.SORRY_RETRY)]
        await call._handle_speech([utterance()])
        assert speech.synthesized[-1] == "We have dosas and idli." and call._failures == 0

    async def test_repeated_failures_end_the_call_politely(self) -> None:
        speech = FakeSpeech(["hi", "hi"])
        call = make_call(FakeWS(), speech)
        scripted_turns(call, [LLMResponseError("unexpected_status"), LLMUnavailableError("timeout")])
        await call._handle_speech([utterance()])
        await call._handle_speech([utterance()])
        assert call.hangup and speech.synthesized[-1] == exotel.speakable(exotel.SORRY_UNAVAILABLE)

    async def test_slow_reply_gets_a_short_filler(self) -> None:
        speech = FakeSpeech(["what's good"])
        call = make_call(FakeWS(), speech, voice_filler_after_ms=20)

        def slow():
            time.sleep(0.1)
            return ("The masala dosa is popular.", False)

        scripted_turns(call, [slow])
        await call._handle_speech([utterance()])
        assert speech.synthesized == [exotel.speakable(exotel.FILLER), "The masala dosa is popular."]

    async def test_long_wait_gets_a_second_filler(self) -> None:
        # E.g. the model waiting out a rate limit: the caller hears "One
        # moment.", then "still working", then the answer, never a long silence.
        speech = FakeSpeech(["what's good"])
        call = make_call(FakeWS(), speech, voice_filler_after_ms=20, voice_still_working_after_ms=80)

        def slow():
            time.sleep(0.2)
            return ("The masala dosa is popular.", False)

        scripted_turns(call, [slow])
        await call._handle_speech([utterance()])
        assert speech.synthesized == [exotel.speakable(exotel.FILLER), exotel.speakable(exotel.STILL_WORKING),
                                      "The masala dosa is popular."]  # fmt: skip

    async def test_quick_reply_gets_no_filler(self) -> None:
        speech = FakeSpeech(["what's good"])
        call = make_call(FakeWS(), speech, voice_filler_after_ms=200, voice_still_working_after_ms=400)
        scripted_turns(call, [("The masala dosa is popular.", False)])
        await call._handle_speech([utterance()])
        assert speech.synthesized == ["The masala dosa is popular."]

    async def test_timings_are_logged_without_content_or_numbers(self, caplog) -> None:
        call = make_call(FakeWS(), FakeSpeech(["two dosas please"]))
        scripted_turns(call, [("Two masala dosas, INR 160.00.", False)])
        with caplog.at_level("INFO", logger="app"):
            await call._handle_speech([utterance()])
        line = next(r.getMessage() for r in caplog.records if "stage=turn" in r.getMessage())
        for field in ("end_of_speech_wait=200ms", "stt=", "agent=", "tts_first_audio=", "total_to_first_audio="):
            assert field in line
        assert "dosa" not in line.lower() and "08000000001" not in line


# ── Voice activity detection ─────────────────────────────────────────────────


class TestUtteranceBuffers:
    def _vad(self) -> UtteranceDetector:
        return UtteranceDetector(sample_rate=RATE, energy_threshold=450, silence_ms=200,
                                 min_speech_ms=100, max_utterance_ms=5000, pre_roll_ms=200)  # fmt: skip

    def test_quiet_start_of_speech_is_kept(self) -> None:
        soft = tone(200, amplitude=300)  # below the threshold: an unvoiced start
        events = self._vad().feed(quiet(400) + soft + tone(300) + quiet(300))
        (utt,) = [e for e in events if e.kind == "utterance"]
        assert utt.audio.startswith(soft)

    def test_utterances_do_not_leak_into_each_other(self) -> None:
        first, second = tone(300, 3000), tone(300, 2000)
        events = self._vad().feed(first + quiet(400) + second + quiet(300))
        a, b = [e.audio for e in events if e.kind == "utterance"]
        assert first not in b and second not in a and second in b
