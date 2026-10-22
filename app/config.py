"""Application configuration loaded from environment variables and .env file."""

import uuid
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level application settings.

    Values are read first from environment variables, then from a local
    ``.env`` file (if present).  Defaults are safe for local development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ── Application identity ─────────────────────────────────────────────────
    app_name: str = "Restaurant AI Voice Agent"
    app_env: str = "development"  # development | staging | production
    debug: bool = True

    # ── Database ─────────────────────────────────────────────────────────────
    # Psycopg 3 URL format: postgresql+psycopg://user:pass@host:port/dbname
    # None → database features are disabled (app still starts for /health).
    database_url: str | None = None

    # ── Connection pool tuning ───────────────────────────────────────────────
    db_pool_size: int = 5
    db_pool_max_overflow: int = 10
    db_connect_timeout: int = 5  # seconds; bounds /health/db when the DB is unreachable
    db_echo_sql: bool = False  # Set True to log generated SQL (never in prod)

    # ── LLM provider (OpenAI-compatible chat completions with tool calling) ──
    # The agent endpoint returns 503 until both an API key and a model are set;
    # the rest of the app works without them.  The model must support tool
    # (function) calling.
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = 30.0
    llm_max_tokens: int = 500
    llm_temperature: float | None = None  # omitted from requests when unset
    # On HTTP 429 with a Retry-After of at most this many seconds, wait and
    # retry once (0 disables).  Longer waits fail fast as "unavailable".
    llm_rate_limit_max_wait_seconds: float = 10.0

    # ── Agent limits ─────────────────────────────────────────────────────────
    agent_max_model_calls: int = 4  # model round-trips per chat request
    agent_max_tool_calls: int = 6  # tool executions per chat request
    agent_session_ttl_seconds: int = 1800
    agent_max_sessions: int = 1000

    # ── Staff authentication ─────────────────────────────────────────────────
    # Sent as the X-Staff-Key header on staff-only endpoints (menu and stock
    # changes, order administration, demo payment simulation).  Unset: those
    # endpoints answer 503 rather than running unauthenticated.
    staff_api_key: SecretStr | None = None
    # HMAC secret for per-order customer tokens (X-Order-Token).  Unset: a
    # random per-process secret is used, so tokens stop working on restart.
    order_token_secret: SecretStr | None = None

    # ── Orders: stock reservations ───────────────────────────────────────────
    # How long a placed, unpaid order holds its stock before it expires.
    order_reservation_ttl_seconds: int = 600
    # How often the app sweeps expired reservations (0 disables the sweep;
    # POST /api/v1/maintenance/expire-reservations still works).
    reservation_sweep_seconds: int = 60

    # ── Exotel telephony (Voicebot applet → WebSocket) ───────────────────────
    # The restaurant that answers calls; a ``restaurant_id`` custom parameter
    # on the applet URL overrides it.  When the username and password are set,
    # the WebSocket requires them as HTTP Basic auth (Exotel sends credentials
    # embedded in the applet URL: wss://user:pass@host/...).
    exotel_restaurant_id: uuid.UUID | None = None
    exotel_ws_username: str | None = None
    exotel_ws_password: SecretStr | None = None
    exotel_sample_rate: int = 8000  # must match the applet's sample-rate (8000 default)
    exotel_greeting: str | None = None  # default: "Welcome to <restaurant>. ..."

    # ── Speech (OpenAI-compatible /audio/transcriptions and /audio/speech) ───
    # Calls are rejected until an API key and both models are set.
    # speech_provider selects the /audio/speech dialect:
    #   openai: response_format "pcm" (raw, tts_sample_rate), no input limit;
    #   groq:   response_format "wav" (rate read from its header, resampled
    #           locally), input split into pieces of at most 200 characters.
    speech_provider: Literal["openai", "groq"] = "openai"
    speech_base_url: str = "https://api.openai.com/v1"
    speech_api_key: SecretStr | None = None
    stt_model: str | None = None  # e.g. whisper-large-v3-turbo, whisper-1
    stt_language: str | None = "en"  # ISO-639-1 hint; unset to auto-detect
    tts_model: str | None = None  # e.g. canopylabs/orpheus-v1-english, tts-1
    tts_voice: str = "alloy"  # Groq Orpheus: e.g. troy, hannah, austin
    tts_sample_rate: int = 24000  # rate of OpenAI's raw "pcm" output
    speech_timeout_seconds: float = 20.0

    # ── Voice activity detection (end of caller's utterance) ─────────────────
    vad_energy_threshold: int = 450  # RMS of 16-bit samples counted as speech
    vad_silence_ms: int = 700  # trailing silence that ends an utterance
    vad_min_speech_ms: int = 200  # shorter bursts are treated as noise
    vad_max_utterance_ms: int = 15000


# Module-level singleton — import this from other modules.
settings = Settings()

