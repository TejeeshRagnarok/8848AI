"""Audio helpers for raw 16-bit little-endian mono PCM ("slin").

Pure functions plus a small energy-based voice activity detector; no external
dependencies, so everything here is cheap to unit-test.
"""

import io
import math
import struct
import wave
from array import array
from dataclasses import dataclass

SAMPLE_WIDTH = 2  # bytes per 16-bit sample

# Exotel requires outbound media chunks that are multiples of 320 bytes,
# between 3.2 KB (100 ms at 8 kHz) and 100 KB.
EXOTEL_CHUNK_MULTIPLE = 320
EXOTEL_MIN_CHUNK = 3200
EXOTEL_MAX_CHUNK = 100_000


def _samples(pcm: bytes) -> array:
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % SAMPLE_WIDTH])
    return samples


def rms(pcm: bytes) -> float:
    """Root-mean-square level of a PCM buffer (0 for empty input)."""
    samples = _samples(pcm)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resampling.  Adequate for telephone-band speech."""
    if src_rate == dst_rate or not pcm:
        return pcm
    samples = _samples(pcm)
    if not samples:
        return b""
    count = max(1, round(len(samples) * dst_rate / src_rate))
    step = src_rate / dst_rate
    last = len(samples) - 1
    out = array("h", bytes(count * SAMPLE_WIDTH))
    for i in range(count):
        pos = i * step
        j = int(pos)
        if j >= last:
            out[i] = samples[last]
        else:
            frac = pos - j
            out[i] = int(samples[j] + (samples[j + 1] - samples[j]) * frac)
    return out.tobytes()


def to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap PCM in a WAV container (for transcription uploads)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def from_wav(data: bytes) -> tuple[bytes, int]:
    """PCM and sample rate from a 16-bit WAV file (stereo is downmixed).

    Raises ``ValueError`` for anything else.  Tolerates the "unknown length"
    data-chunk size that streaming TTS endpoints write.
    """
    try:
        with wave.open(io.BytesIO(data)) as wav:
            channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
            pcm = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError("not a PCM WAV file") from exc
    if width != SAMPLE_WIDTH or channels not in (1, 2):
        raise ValueError("unsupported WAV format")
    if channels == 2:
        stereo = _samples(pcm)
        pcm = array("h", ((stereo[i] + stereo[i + 1]) // 2
                          for i in range(0, len(stereo) - 1, 2))).tobytes()  # fmt: skip
    return pcm, rate


def exotel_chunks(pcm: bytes, chunk_size: int = 8000) -> list[bytes]:
    """Split PCM into chunks Exotel accepts: each a multiple of 320 bytes and
    at least 3.2 KB.  The tail is padded with silence."""
    size = min(max(chunk_size, EXOTEL_MIN_CHUNK), EXOTEL_MAX_CHUNK)
    size -= size % EXOTEL_CHUNK_MULTIPLE
    chunks = [pcm[i : i + size] for i in range(0, len(pcm), size)]
    if chunks:
        tail = chunks[-1]
        padded = max(EXOTEL_MIN_CHUNK, -(-len(tail) // EXOTEL_CHUNK_MULTIPLE) * EXOTEL_CHUNK_MULTIPLE)
        chunks[-1] = tail + b"\x00" * (padded - len(tail))
    return chunks


def silence(ms: int, sample_rate: int) -> bytes:
    return struct.pack("<h", 0) * (sample_rate * ms // 1000)


# ── Voice activity detection ─────────────────────────────────────────────────


@dataclass
class VadEvent:
    kind: str  # "speech_start" | "utterance"
    audio: bytes = b""


class UtteranceDetector:
    """Energy-based endpointing over 20 ms frames.

    ``feed`` returns ``speech_start`` when the caller begins talking (used for
    barge-in) and ``utterance`` with the buffered audio once they stop for
    ``silence_ms`` (or hit ``max_utterance_ms``).  Bursts shorter than
    ``min_speech_ms`` are discarded as noise.
    """

    FRAME_MS = 20

    def __init__(
        self,
        *,
        sample_rate: int,
        energy_threshold: int,
        silence_ms: int,
        min_speech_ms: int,
        max_utterance_ms: int,
    ) -> None:
        self._frame_bytes = sample_rate * self.FRAME_MS // 1000 * SAMPLE_WIDTH
        self._threshold = energy_threshold
        self._silence_frames = max(1, silence_ms // self.FRAME_MS)
        self._min_speech_frames = max(1, min_speech_ms // self.FRAME_MS)
        self._max_frames = max(1, max_utterance_ms // self.FRAME_MS)
        self._pending = b""
        self.reset()

    def reset(self) -> None:
        self._buffer = bytearray()
        self._in_speech = False
        self._speech_frames = 0
        self._silent_run = 0
        self._frames = 0

    def feed(self, pcm: bytes) -> list[VadEvent]:
        events: list[VadEvent] = []
        data = self._pending + pcm
        usable = len(data) - len(data) % self._frame_bytes
        self._pending = data[usable:]
        for i in range(0, usable, self._frame_bytes):
            event = self._frame(data[i : i + self._frame_bytes])
            if event is not None:
                events.append(event)
        return events

    def _frame(self, frame: bytes) -> VadEvent | None:
        loud = rms(frame) >= self._threshold
        if not self._in_speech:
            if not loud:
                return None
            self._in_speech = True
            self._buffer = bytearray(frame)
            self._speech_frames, self._silent_run, self._frames = 1, 0, 1
            return VadEvent("speech_start") if self._min_speech_frames == 1 else None

        self._buffer += frame
        self._frames += 1
        if loud:
            self._speech_frames += 1
            self._silent_run = 0
            if self._speech_frames == self._min_speech_frames:
                return VadEvent("speech_start")
        else:
            self._silent_run += 1

        ended = self._silent_run >= self._silence_frames or self._frames >= self._max_frames
        if not ended:
            return None
        enough = self._speech_frames >= self._min_speech_frames
        audio = bytes(self._buffer)
        self.reset()
        return VadEvent("utterance", audio) if enough else None
