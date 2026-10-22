"""Speech-to-text and text-to-speech behind small protocols.

The default implementation calls OpenAI-compatible ``/audio/transcriptions``
and ``/audio/speech`` endpoints.  Other providers (Deepgram, ElevenLabs, ...)
can be added by implementing ``SpeechToText`` / ``TextToSpeech``.

As in ``app.agent.llm``, error messages are fixed strings: provider
responses, URLs and keys never reach callers or logs.
"""

import logging
import re
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from app.config import Settings, settings
from app.telephony.audio import from_wav, resample, to_wav

logger = logging.getLogger(__name__)

GROQ_TTS_MAX_CHARS = 200  # Groq Orpheus input limit per request


_MONEY = re.compile(r"\bINR\s?([\d,]+)\.(\d{2})\b")
_ORDER_NUMBER = re.compile(r"\bORD[-‐‑]?(\d+)\b")


def _rupees(match: re.Match) -> str:
    rupees, paise = match.group(1).replace(",", ""), match.group(2)
    words = f"{int(rupees)} rupees"
    return words if paise == "00" else f"{words} {int(paise)} paise"


def speakable(text: str) -> str:
    """Turn a chat-style reply into something a phone voice can read well:
    no markdown, prices as rupees, order numbers digit by digit."""
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|#{1,6})\s+", "", raw)  # bullets and headings
        line = re.sub(r"(\*\*|__|`)", "", line).strip()  # emphasis and code marks
        if not line:
            continue
        if line[-1] not in ".!?:,":
            line += "."
        lines.append(line)
    spoken = " ".join(lines)
    spoken = spoken.replace("×", " ").replace("–", ",").replace("—", ",")
    spoken = _MONEY.sub(_rupees, spoken)
    spoken = _ORDER_NUMBER.sub(lambda m: "order " + " ".join(m.group(1)), spoken)
    spoken = re.sub(r"\s+([.,!?])", r"\1", spoken)
    return re.sub(r"\s{2,}", " ", spoken).strip()


def split_text(text: str, limit: int) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` characters, preferring
    sentence, then word boundaries."""
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit + 1)
            cut = cut if cut > 0 else limit
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if pieces and len(pieces[-1]) + 1 + len(sentence) <= limit:
            pieces[-1] = f"{pieces[-1]} {sentence}"
        else:
            pieces.append(sentence)
    return pieces


class SpeechError(Exception):
    """Base class; ``str(exc)`` is always safe to show."""


class SpeechNotConfiguredError(SpeechError):
    def __init__(self) -> None:
        super().__init__("Speech services are not configured.")


class SpeechUnavailableError(SpeechError):
    def __init__(self, reason: str) -> None:
        super().__init__("Speech services are temporarily unavailable.")
        self.reason = reason  # short internal category, safe to log


class SpeechToText(Protocol):
    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """Text spoken in ``pcm`` (may be empty).  Raises ``SpeechError`` only."""
        ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str, sample_rate: int) -> bytes:
        """16-bit mono PCM at ``sample_rate``.  Raises ``SpeechError`` only."""
        ...


class OpenAICompatibleSpeech:
    """Both directions through one OpenAI-compatible audio API."""

    def __init__(self, config: Settings, client: httpx.Client | None = None) -> None:
        if config.speech_api_key is None or not config.stt_model or not config.tts_model:
            raise SpeechNotConfiguredError()
        base = config.speech_base_url.rstrip("/")
        self._stt_url = base + "/audio/transcriptions"
        self._tts_url = base + "/audio/speech"
        self._host = urlsplit(base).hostname or "unknown"  # the only part ever logged
        self._provider = config.speech_provider
        self._stt_model = config.stt_model
        self._language = config.stt_language
        self._tts_model = config.tts_model
        self._voice = config.tts_voice
        self._tts_rate = config.tts_sample_rate
        self._client = client or httpx.Client(timeout=httpx.Timeout(config.speech_timeout_seconds))
        self._headers = {"Authorization": f"Bearer {config.speech_api_key.get_secret_value()}"}

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        data = {"model": self._stt_model, "response_format": "json"}
        if self._language:
            data["language"] = self._language
        files = {"file": ("utterance.wav", to_wav(pcm, sample_rate), "audio/wav")}
        response = self._post(self._stt_url, data=data, files=files)
        try:
            text = response.json()["text"]
        except (ValueError, KeyError, TypeError):
            self._log("invalid_transcription", response.status_code)
            raise SpeechUnavailableError("invalid_transcription") from None
        if not isinstance(text, str):
            raise SpeechUnavailableError("invalid_transcription")
        return text.strip()

    def synthesize(self, text: str, sample_rate: int) -> bytes:
        if self._provider == "groq":
            pieces = split_text(text, GROQ_TTS_MAX_CHARS)
            return b"".join(self._synthesize_wav(piece, sample_rate) for piece in pieces)
        body = {
            "model": self._tts_model,
            "voice": self._voice,
            "input": text,
            "response_format": "pcm",  # raw 16-bit mono at the provider's rate
        }
        response = self._post(self._tts_url, json=body)
        return resample(response.content, self._tts_rate, sample_rate)

    def _synthesize_wav(self, text: str, sample_rate: int) -> bytes:
        # No "sample_rate" field: Groq Orpheus always renders 24 kHz and that
        # field only relabels the WAV header (so 8000 plays 3x too slow).  The
        # default header is truthful; resample locally instead.
        body = {
            "model": self._tts_model,
            "voice": self._voice,
            "input": text,
            "response_format": "wav",
        }
        response = self._post(self._tts_url, json=body)
        try:
            pcm, rate = from_wav(response.content)
        except ValueError:
            self._log("invalid_audio", response.status_code)
            raise SpeechUnavailableError("invalid_audio") from None
        return resample(pcm, rate, sample_rate)

    def _post(self, url: str, **kwargs) -> httpx.Response:
        try:
            response = self._client.post(url, headers=self._headers, **kwargs)
        except httpx.TimeoutException:
            self._log("timeout")
            raise SpeechUnavailableError("timeout") from None
        except httpx.HTTPError:
            self._log("connection_error")
            raise SpeechUnavailableError("connection_error") from None
        if response.status_code != 200:
            reason = {401: "authentication_failed", 403: "authentication_failed",
                      429: "rate_limited"}.get(response.status_code, "provider_error")  # fmt: skip
            self._log(reason, response.status_code)
            raise SpeechUnavailableError(reason)
        return response

    def _log(self, reason: str, status: int | None = None) -> None:
        logger.warning("Speech request to %s failed: %s (status %s)", self._host, reason, status)


_speech: OpenAICompatibleSpeech | None = None


def get_speech() -> OpenAICompatibleSpeech:
    """The configured speech provider (reusing one HTTP client).

    Raises ``SpeechNotConfiguredError`` when the key or a model is missing.
    """
    global _speech  # noqa: PLW0603
    if _speech is None:
        _speech = OpenAICompatibleSpeech(settings)
    return _speech
