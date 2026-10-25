# Restaurant AI Voice Agent

An AI-powered voice ordering agent for restaurants. Customers call an inbound phone number, talk to the AI, ask about the menu, build and change their order, and confirm it, with no human on the line. Staff follow orders and stock live on a dashboard.

---

## Demo

> **Video coming soon.** A real phone call: ordering, changing the cart, interrupting the bot,
> an out-of-stock item with an alternative, confirming, and the order appearing live on the
> staff dashboard.

<!-- Replace the line above with the video, e.g.
[![Watch the demo](docs/demo-thumbnail.png)](https://youtu.be/<video-id>)
or drag the .mp4 into GitHub's README editor. -->

### How a call works

```mermaid
flowchart LR
    caller([Caller's phone]) <-->|PSTN| exotel[Exotel Voicebot]
    exotel <-->|"WebSocket, 8 kHz PCM"| ws[FastAPI call handler]
    ws --> vad["Voice activity detection<br/>700 ms end of speech, barge-in"]
    vad --> stt["Speech to text<br/>Whisper large-v3-turbo"]
    stt --> agent["LLM agent<br/>Llama 3.3 70B, 5 tools"]
    agent <-->|"live menu, stock, cart"| db[(PostgreSQL)]
    agent --> guard["Output guardrail<br/>no unverified claims"]
    guard --> tts["Text to speech<br/>Orpheus, streamed in pieces"]
    tts -->|"200 ms audio chunks"| ws
    agent -.->|"proposal only"| confirm{"Caller says yes<br/>or presses 1"}
    confirm -->|"server places order:<br/>row locks, idempotency"| db
    db <-->|"Server-Sent Events"| dash[Staff dashboard]
```

### Highlights

- **The model never places orders.** It can only propose a cart; the server places it after an
  explicit "yes" (or keypress 1), re-checking prices and stock under row locks.
- **No hallucinated facts reach the caller.** Replies claiming an order was placed or a payment
  was taken are replaced unless the database confirmed it in the same turn; tool output is
  treated as data, never as instructions.
- **Built for real phone latency.** A fresh menu and cart snapshot in every prompt means most
  turns need one model call; replies are synthesised and streamed piece by piece; cached
  phrases start the greeting in about 0.1 s; fillers ("One moment.") cover slow turns.
- **A caller is never left in silence.** Every stage has a timeout, rate limits are waited out
  once, failures get a spoken apology, and two failures in a row end the call politely.
- **Stock can never be oversold.** Reservations with expiry, idempotent orders, ordered row
  locks and database constraints, covered by concurrency tests against real PostgreSQL.
- **755 automated tests** (unit, mocked, live PostgreSQL and concurrency races).

**Stack:** Python 3.13, FastAPI, WebSockets, SQLAlchemy 2, Alembic, PostgreSQL, Groq
(Llama 3.3 70B, Whisper, Orpheus), Exotel, pytest.

---

## Prerequisites

| Requirement | Version | Details |
|---|---|---|
| Python | 3.13 or later | Local runtime environment |
| Git | any recent version | Version control |
| PostgreSQL | 18.6 | Running locally on port `5432` |
| Database | `restaurant_ai` | Application database |
| Application User | `restaurant_app` | Dedicated non-superuser role |

---

## Quick Start

### 1. Clone and enter the project

```powershell
git clone <repo-url> restaurant_voice_ai_agent
cd restaurant_voice_ai_agent
```

### 2. Create the virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

> **Note:** If PowerShell blocks script execution, run once:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

### 4. Install dependencies

```powershell
pip install -e ".[dev]"
```

Dependencies include:
- **FastAPI** + **Uvicorn**: Core web framework and ASGI server
- **SQLAlchemy 2.x**: ORM, connection pool, and query engine
- **Psycopg 3 (`psycopg[binary]`)**: PostgreSQL driver for Python / Windows
- **Alembic**: Database schema migrations
- **Pydantic Settings**: Type-safe configuration management
- **pytest** + **pytest-asyncio** + **httpx**: Test framework

### 5. Configure your local environment (`.env`)

Copy the template file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` manually and configure your database credentials:

```dotenv
DATABASE_URL=postgresql+psycopg://restaurant_app:<password>@localhost:5432/restaurant_ai
```

> [!IMPORTANT]
> - Replace `<password>` with the actual password for the `restaurant_app` role.
> - If your password contains special characters (such as `@`, `:`, `/`, `#`), percent-encode them (e.g., `p@ss#1` becomes `p%40ss%231`).
> - **Never commit `.env` to Git.** It is excluded via `.gitignore`.

---

## Database Migrations (Alembic)

Alembic manages schema migrations. Migrations read `DATABASE_URL` directly from application settings (`app.config.settings`) and are **never** executed automatically during application startup.

### Run migrations to the latest revision

```powershell
alembic upgrade head
```

### View current revision status

```powershell
alembic current
```

### View migration history

```powershell
alembic history --verbose
```

### Rollback one revision

```powershell
alembic downgrade -1
```

### Generate a new autogenerated migration (Task 3+)

```powershell
alembic revision --autogenerate -m "create_menu_and_order_tables"
```

---

## Running the Application

Start the local development server:

```powershell
python -m uvicorn app.main:app --reload
```

The API endpoints will be available at:
- **Application root:** http://127.0.0.1:8000
- **Liveness probe (`/health`):** http://127.0.0.1:8000/health
  - Returns `200 OK` when the process is running.
  - Does **not** require PostgreSQL or `DATABASE_URL`.
- **Database readiness probe (`/health/db`):** http://127.0.0.1:8000/health/db
  - Executes `SELECT 1` against PostgreSQL.
  - Returns `200 OK` (`{"status": "ok", "database": "connected"}`) when reachable.
  - Returns `503 Service Unavailable` (`{"status": "error", "database": "unreachable"}`) if the database is unreachable or unconfigured, without leaking credentials or stack traces.
- **Interactive OpenAPI docs:** http://127.0.0.1:8000/docs *(development mode only)*

### Authentication

| Who | How | Can |
|---|---|---|
| Anyone | — | Read restaurants and menus (with live stock), place orders, use the agent chat and the Exotel socket |
| The order's customer | `X-Order-Token: <customer_token>` (returned when the order is placed) | Track (`GET /orders/{id}/track`, no contact details) and cancel (`POST /orders/{id}/cancel`) **that one order** |
| Staff | `X-Staff-Key: <STAFF_API_KEY>` | Everything else: menu/restaurant changes, stock, order reads and status changes, demo payments, maintenance |

Staff endpoints answer `401` without a valid key and `503` while `STAFF_API_KEY` is
unset (they never run unauthenticated). In `/docs`, put the header value into each
staff request's `x-staff-key` field. The agent chat API remains unauthenticated
(development only): do not expose this service publicly.

### API overview (`/api/v1`)

| Method | Path | Access | Purpose |
|---|---|---|---|
| `POST` | `/restaurants` | staff | Create a restaurant (`409` on duplicate slug) |
| `GET` | `/restaurants`, `/restaurants/{id}` | public | List / fetch restaurants |
| `PATCH`, `DELETE` | `/restaurants/{id}` | staff | Update / deactivate (soft delete) |
| `POST` | `/restaurants/{id}/menu-items` | staff | Add a menu item |
| `GET` | `/restaurants/{id}/menu-items?is_available=&in_stock=` | public | Menu with `available_quantity` and `orderable` |
| `GET` | `/menu-items/{item_id}` | public | One item |
| `PATCH`, `DELETE` | `/menu-items/{item_id}` | staff | Update / delete (`409` while stock is reserved for it) |
| `GET` | `/restaurants/{id}/inventory` | staff | On hand, reserved, available and sold per item |
| `PUT` | `/menu-items/{item_id}/inventory` | staff | Set on-hand count (`{"on_hand": 40}`); starts tracking |
| `GET` | `/menu-items/{item_id}/inventory/movements` | staff | The item's stock ledger |
| `POST` | `/restaurants/{id}/orders` | public | Place an order: reserves stock, creates the demo payment |
| `GET` | `/orders/{id}/track` | customer token | Status, items, total, payment status |
| `POST` | `/orders/{id}/cancel` | customer token / staff | Cancel (idempotent) |
| `GET` | `/restaurants/{id}/orders?status=`, `/orders/{id}` | staff | Orders with contact details and payment |
| `PATCH` | `/orders/{id}/status` | staff | `preparing`, `ready`, `completed`, `cancelled` |
| `GET` | `/orders/{id}/payment`, `/payments/{payment_id}` | staff | Payment record |
| `POST` | `/payments/{payment_id}/simulate` | staff | **Demo only**: simulate `succeeded` / `failed` |
| `POST` | `/maintenance/expire-reservations` | staff | Expire lapsed reservations now |

Prices are integer paise (`12000` = ₹120.00). Lists return
`{"items", "total", "limit", "offset"}` with `limit` ≤ 100. Errors use
`{"detail": ...}` and never include SQL, URLs or stack traces.

### Orders, stock and payments

**Pricing.** Clients send only `menu_item_id` + `quantity` per line (1–100, at most
50 lines, each item once). The server reads current prices from PostgreSQL, locking
the rows for the transaction, and computes `line_total = unit_price × quantity` and
`total = subtotal = Σ line totals` (no taxes or fees yet). Any client-supplied price,
total or status is rejected with `422`. Item name and unit price are snapshotted onto
each line, so later menu edits or deletions never change past orders.

**Stock model.** An item is stock-tracked once it has an inventory row (`PUT
.../inventory`); untracked items are limited only by `is_available`.

| Field | Meaning |
|---|---|
| `on_hand` | Physical units not yet sold |
| `reserved` | Units held for placed-but-unpaid orders (part of `on_hand`) |
| `available` | `on_hand − reserved`: what new orders can take |
| `sold` | Units sold through paid orders (reporting) |

An item is `orderable` when `is_available` is true and, if tracked, `available > 0`.
The database enforces `0 ≤ reserved ≤ on_hand`, so it can never be oversold, and
every change is recorded in `inventory_movements`, where a unique
`(order_id, menu_item_id, kind)` makes each order movement happen at most once.

**Order and stock lifecycle.**

```
cart (agent proposal, nothing saved or reserved)
  └─ customer confirms ─► pending      stock reserved, demo payment pending,
                            │           reservation_expires_at = now + ORDER_RESERVATION_TTL_SECONDS
                            ├─ payment succeeded ─► confirmed   reserved → sold (on_hand −q, sold +q)
                            │                          └─► preparing ─► ready ─► completed  (history kept)
                            ├─ payment failed ─────► cancelled (payment_failed)  reservation released
                            ├─ reservation lapses ─► cancelled (expired)         reservation released
                            └─ customer/staff ─────► cancelled (customer/staff)  reservation released
confirmed ─ cancel ─► cancelled: sold stock returned, payment refunded (demo)
preparing/ready ─ staff cancel ─► cancelled: stock stays sold (food used), payment refunded (demo)
completed ─► final; never deleted, never cancellable
```

`orders.inventory_state` (`none` · `reserved` · `sold` · `released` · `returned`)
records which movement an order holds, so repeated or concurrent cancellations,
payments and expiry runs apply each effect once. Every change locks the order row
first (then menu rows, then inventory rows, each in key order), so concurrent
requests serialize without deadlocks. A failed step rolls back the whole
transaction: no partial orders, reservations or payments.

**Demo payments (DEMO PAYMENT MODE).** No gateway, no card or UPI details, no real
money. Each order gets exactly one payment (`UNIQUE(order_id)`, `mode = 'demo'`)
for exactly the order total. Its status changes only through
`POST /payments/{payment_id}/simulate` (staff) or the cancellation/expiry rules:

```
pending ─► succeeded ─► refunded (paid order cancelled)
   ├─────► failed      (simulated decline)
   ├─────► expired     (reservation lapsed)
   └─────► cancelled   (order cancelled before payment)
```

`amount_paise` must equal the payment amount (`422 amount_mismatch` otherwise);
repeating the same outcome returns the payment with `Idempotent-Replayed: true` and
no side effects; a final payment cannot change outcome (`409`); paying after the
reservation lapsed expires the order instead (`409`). Every payment response carries
`"simulated": true` and a demo notice. A customer saying "I paid" changes nothing.

**Reservation expiry.** The app sweeps lapsed reservations every
`RESERVATION_SWEEP_SECONDS` (default 60; a background task inside the app process,
using `SELECT … FOR UPDATE SKIP LOCKED`, safe with several processes) and on demand
via `POST /api/v1/maintenance/expire-reservations`. Paying a lapsed order also
expires it on the spot.

**Errors.** `404` unknown restaurant/order (also for another customer's order) ·
`409` restaurant inactive, idempotency conflict, not enough stock, disallowed status
change, or payment already final · `422` invalid body, missing/foreign/unavailable
items, amount mismatch.

**Idempotency.** Send an `idempotency_key` (unique per restaurant) and reuse it when
retrying. Same key + same request → `200` with the original order and header
`Idempotent-Replayed: true` (stock is reserved once). Same key + different request →
`409`. If omitted, the server generates a key and retries cannot be deduplicated.

### Ordering assistant (LLM agent)

`POST /api/v1/restaurants/{restaurant_id}/agent/chat`

> **Development only.** No authentication, no authorization, and no session
> ownership checks (anyone with a `session_id` can continue that conversation).
> Sessions live in process memory: run a single worker, and expect them to vanish
> on restart. Do not expose publicly.

Requires an OpenAI-compatible chat-completions provider whose model supports
**tool/function calling**. Set `LLM_API_KEY` and `LLM_MODEL` (and `LLM_BASE_URL` for
providers other than OpenAI) in `.env`. Without them this endpoint returns `503` and
the rest of the API works normally. Automated tests never call a real provider.

```jsonc
// request
{"message": "Two masala dosas please", "session_id": "<from previous reply, optional>"}
// response
{"reply": "...", "session_id": "...", "requires_confirmation": true,
 "pending_order": {"proposal_id": "...", "items": [...], "total_paise": 24100, "total": "INR 241.00", ...},
 "order": null, "order_replayed": false}
```

The model can only use five tools, all bound to the restaurant in the URL:
`get_menu` (search/category; only orderable items, with live `available_quantity`),
`get_menu_item`, `create_order`, `get_order_status` (orders placed in *this*
conversation only, read from PostgreSQL), and `get_popular_items` (most ordered
dishes from paid orders, or a note that there is too little history to say).

**Orders are never placed by the model.** `create_order` only prices and
stock-checks the request through the order engine (nothing is reserved) and stores a
server-side *proposal*; adding, removing or changing items replaces the proposal.
Items that are missing, switched off or short of stock come back as problems with
the available quantity and up to five in-stock *alternatives* from the database. To place it, the client
sends `{"session_id": ..., "confirm_proposal_id": "<pending_order.proposal_id>"}`
once the customer explicitly agrees. That confirmation is handled by the server
alone (no model call): it places the stored proposal with a stable idempotency key,
re-checking prices, availability and stock under row locks, and reserving the
stock. If prices changed, nothing is placed and the response carries an updated
proposal to confirm again; if stock ran out, the reply names the item and quantity.
The placement reply (written by the server) states the reservation time and that the
demo payment is pending. Retrying a
confirmation returns the same order (`order_replayed: true`).

Model replies that claim an order was placed, a payment was taken, a delivery was
arranged, or a WhatsApp/SMS was sent are replaced before reaching the customer,
unless `get_order_status` read that very fact from the database in the same turn
(e.g. a payment claim is allowed only when the stored payment is `succeeded`). A
replaced reply states the stored status instead. A customer saying "I paid" never
changes a payment.

Errors: `404` unknown restaurant/session · `409` inactive restaurant, busy session,
or nothing matching to confirm · `422` invalid body · `502` malformed provider
response · `503` assistant not configured or provider unavailable.

### Phone calls (Exotel Voicebot applet)

`WebSocket /telephony/exotel/voicebot`

Exotel streams each call's audio (16-bit mono PCM, 8 kHz) over this socket. The
server detects when the caller stops speaking, transcribes the utterance, runs it
through the ordering assistant above (one chat turn, with the caller's number given
to the model as context), and streams the spoken reply back. Talking over the bot
stops its playback (`clear`).

Orders are still placed only by the server-side confirmation: when a proposal is
pending, a short, clear "yes" (or pressing **1**) sends `confirm_proposal_id`.
Anything else ("yes, but add a lassi") goes to the model as a normal turn.

Setup:

1. In `.env` set `LLM_*`, `SPEECH_API_KEY`, `STT_MODEL`, `TTS_MODEL`,
   `EXOTEL_RESTAURANT_ID`, `EXOTEL_WS_USERNAME` and `EXOTEL_WS_PASSWORD` (see
   `.env.example`). Calls are refused until speech is configured.
2. Expose the server publicly over HTTPS (for example `ngrok http 8000`).
3. In Exotel → App Bazaar → your flow, add a **Voicebot** applet with the URL
   `wss://<user>:<password>@<public-host>/telephony/exotel/voicebot`
   (append `?restaurant_id=<uuid>` to override the default restaurant), save, and
   assign the flow to your ExoPhone.

Speech uses an OpenAI-compatible audio API (`/audio/transcriptions`,
`/audio/speech`). `SPEECH_PROVIDER=openai` requests raw `pcm`;
`SPEECH_PROVIDER=groq` requests `wav` at the call's sample rate and splits replies
into ≤200-character pieces (Groq Orpheus limit). Other providers can implement the
`SpeechToText` / `TextToSpeech` protocols in `app/telephony/speech.py`.

**How a call turn works (latency and interruptions).**

- *End of speech*: the caller's turn ends after `VAD_SILENCE_MS` (700 ms) of silence;
  200 ms of audio before speech is detected is kept so first syllables are not lost.
  Utterances queued together are transcribed and answered as one turn.
- *Agent*: every turn sends the model a fresh database snapshot (menu with prices
  and stock for menus up to 40 items, and the current cart), so most questions need
  one model call. History keeps only what was said, not tool output. Descriptions
  come from `get_menu` (tool results are treated as data, never as instructions).
- *Speech*: replies are synthesised piece by piece and each piece is sent as soon as
  it is ready. Fixed phrases (greeting, apologies, "One moment.") are synthesised once
  per process and reused. If a reply takes longer than `VOICE_FILLER_AFTER_MS`, the
  caller hears "One moment.", and after `VOICE_STILL_WORKING_AFTER_MS` a cached "still
  working" phrase (e.g. while the model waits out a rate limit).
- *Barge-in*: when the caller starts talking while the bot is speaking, the server
  sends Exotel's `clear` (drops queued audio) and stops sending the rest of the reply.
  Audio is sent in 200 ms chunks, so playback stops within about one chunk plus network
  delay; Exotel cannot cut a chunk that has started playing.
- *Stale replies*: if the caller says something new while a reply is being prepared,
  that reply is not spoken and is removed from the history; one turn answers
  everything the caller said. A reply that confirms a placed order is always spoken.
- *Failures*: a provider outage or rate limit gets a spoken apology and the call
  continues; the call ends only after two failures in a row. A malformed tool call
  from the model is retried once. Each turn logs
  `end_of_speech_wait / stt / agent / tts_first_audio / total_to_first_audio` timings
  (no transcript, no phone numbers).

**Reliability: a caller is never left in silence.**

- *Fixed phrases on disk*: the greeting and the apologies/goodbye are synthesised once
  and stored in `VOICE_PHRASE_CACHE_DIR` (e.g. `.cache/voice_phrases`, git-ignored); the
  app prepares missing ones at startup and retries every 2 minutes for an hour if the
  TTS quota is exhausted. The greeting then starts within ~0.1 s of Exotel's `start`.
- *Bounded stages*: speech requests time out after `SPEECH_TIMEOUT_SECONDS` (8 s), a whole
  assistant turn after `VOICE_TURN_TIMEOUT_SECONDS` (30 s, room for one rate-limit pause); a socket that never sends
  `start` is closed after `VOICE_START_TIMEOUT_SECONDS`, a call whose audio stream stops
  after `VOICE_IDLE_TIMEOUT_SECONDS`.
- *Every turn ends in a terminal state*: answered; recovered with a short cached apology
  (speech not understood, assistant slow or failing, reply audio could not be generated,
  any unexpected error); or, after two failed turns in a row, a spoken goodbye and the
  call is closed. If not even cached audio can play, the call is closed rather than left
  silent.
- *Logs*: one line per stage, e.g.
  `voice call=<sid> turn=3 stage=tts result=failed duration=0.05s error=rate_limited audio_bytes=0`,
  plus a per-turn summary and a per-call summary (`reason`, `duration`, `turns`,
  `audio_bytes`). Timestamps are included; phone numbers, keys, audio and what the caller
  said are never logged.
- *Keep the host awake*: while the server runs on a laptop, its sleep/Modern Standby makes
  calls wait or fail until it wakes. Disable sleep while testing, or host the server on an
  always-on machine.

**MVP configuration: Groq for everything, one API key.** LLM
`llama-3.3-70b-versatile`, STT `whisper-large-v3-turbo`, TTS
`canopylabs/orpheus-v1-english` (voice e.g. `troy`), all at
`https://api.groq.com/openai/v1`. Note:

- Orpheus TTS is a Groq **preview** model (may change or be withdrawn) and is
  **English only**. On the free tier it allows **3,600 characters per day** (and
  1,200 per minute, 100 requests per day): roughly 25-30 short spoken replies a day
  across all calls. When it runs out the bot can only play cached phrases. Real use
  needs Groq's paid tier or another TTS provider.
- Groq bills each transcription as at least 10 seconds, and each caller
  utterance is one request.

**Language limitation.** The phone agent is English-only today: the system prompt
asks for English replies, `STT_LANGUAGE=en` forces English transcription, and
Orpheus can only speak English. English TTS is **not** suitable for Telugu or
Hindi callers. Whisper can transcribe Hindi (`hi`) and Telugu (`te`), but
multilingual calls also need a TTS that speaks those languages (for example
Sarvam AI Bulbul, Google Cloud Text-to-Speech or Azure AI Speech, which all have
Hindi and Telugu voices), a prompt change to reply in the caller's language, and
an LLM tested in those languages.

> **Development only.** Sessions are in memory (run one worker). Always set the
> WebSocket username and password before exposing the server.

---

## Restaurant Dashboard

Open **http://127.0.0.1:8000/dashboard** (or `https://<your-ngrok-host>/dashboard`) and
sign in with `STAFF_API_KEY` (optionally a restaurant slug; default is
`EXOTEL_RESTAURANT_ID`). Two sections:

- **Live Orders**: every channel (AI voice, AI chat, walk-in/POS, API; older orders show
  "Unknown"), with search (order number, name, phone digits), status/channel filters,
  order details (lines, prices, total, contact number or "Not provided", payment and
  reservation expiry) and actions: mark paid (demo), start preparing, mark ready,
  complete, cancel. **Record walk-in sale** creates a `WALK_IN_POS` order priced and
  stock-checked by the server (contact number optional).
- **Menu & Inventory**: price, sellable quantity, on hand, reserved, sold and status per
  dish; add items (with optional starting stock), edit, adjust stock up or down with a
  reason (stock ledger), switch on/off, and retire/restore (retiring never deletes:
  past orders and the ledger keep pointing at the dish).

**One source of truth.** The dashboard, the voice/chat agent and the public API use the
same services and PostgreSQL rows. The agent reads the live menu and stock on every
turn, so a price change, stock count or retired dish applies to the very next caller;
orders reserve stock under row locks, so a counter sale and a phone order can never
both take the last unit. **Every counter sale must be recorded** (walk-in sale here, or a
future POS integration): unrecorded sales cannot be reflected in stock. No POS system is
integrated yet.

**Live updates** use Server-Sent Events (`/api/v1/dashboard/events`): the server runs one
small aggregate query every `DASHBOARD_EVENTS_SECONDS` (3 s) per open dashboard and
tells the browser to refresh when anything changed; if the stream drops, the page polls
every 15 s.

**Security.** Signing in sets a signed, `HttpOnly`, `SameSite=Strict` session cookie
bound to one restaurant (12 h, `DASHBOARD_SESSION_HOURS`); the staff key is not stored
in the browser, and rotating `STAFF_API_KEY` signs everyone out. Every dashboard
endpoint takes the restaurant from the session, never from the request; changes also
need an `X-Requested-With: dashboard` header (blocks cross-site forms). Five failed
sign-ins per client per 5 minutes are allowed. Prices, totals, payment state and stock
are never accepted from the browser. Payments remain **demo only**.

| Method | Path (`/api/v1/dashboard`) | Purpose |
|---|---|---|
| `POST`/`GET`/`DELETE` | `/session` | Sign in / who am I / sign out |
| `GET` | `/summary` | Active, today's, awaiting payment, low and out of stock |
| `GET` | `/events` | Live change stream (SSE) |
| `GET` | `/orders?status=&source=&active_only=&q=` | Orders of this restaurant |
| `GET` | `/orders/{id}` | One order |
| `POST` | `/orders/{id}/status`, `/orders/{id}/cancel`, `/orders/{id}/mark-paid` | Order actions |
| `POST` | `/orders/walk-in` | Record a counter sale |
| `GET`/`POST` | `/menu` | Menu with stock / add item |
| `PATCH` | `/menu/{id}` | Edit item, switch on/off |
| `POST` | `/menu/{id}/stock` | Stock change with reason |
| `POST` | `/menu/{id}/deactivate`, `/menu/{id}/activate` | Retire / restore |
| `GET` | `/menu/{id}/movements` | Stock ledger |

---

## Demo Walkthrough (empty database → completed, paid order)

> Everything below is **DEMO PAYMENT MODE**: payments are simulated and no real money
> is ever charged or refunded.

### 1. Database, settings and demo data

```powershell
alembic upgrade head                 # creates/updates all tables (incl. inventory and payments)
python -m app.demo_seed              # creates "Spice Garden Demo", 8 items and their stock
```

The seed prints `EXOTEL_RESTAURANT_ID=<uuid>`; put it in `.env`. It is safe to re-run:
the restaurant is found by slug (`spice-garden-demo`) and items by name, and stock
is set only for items not yet tracked. `python -m app.demo_seed --reset-stock`
resets on-hand counts to the demo levels (never below units reserved for unpaid
orders). Also set `STAFF_API_KEY` (any long random string) and `ORDER_TOKEN_SECRET`
in `.env`.

| Item | Category | Price | Demo stock |
|---|---|---|---|
| Idli | Breakfast | ₹40 | 60 |
| Plain Dosa | Dosa | ₹60 | 40 |
| Masala Dosa | Dosa | ₹80 | 30 |
| Medu Vada | Breakfast | ₹50 | 40 |
| Upma | Breakfast | ₹50 | 20 |
| Veg Meals | Meals | ₹150 | 15 |
| Tea | Beverages | ₹20 | 100 |
| Filter Coffee | Beverages | ₹25 | 100 |

To reset a development database completely: `alembic downgrade base`,
`alembic upgrade head`, then `python -m app.demo_seed` (this deletes all data).

### 2. Start the app

```powershell
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs. For staff requests, fill the `x-staff-key` field
with your `STAFF_API_KEY`.

### 3. Order by conversation (text)

`POST /api/v1/restaurants/{restaurant_id}/agent/chat`:

1. `{"message": "I want two masala dosas and one filter coffee. My number is +91 98765 43210."}`
   → the reply reads back the items and total; `pending_order` holds the
   proposal. Nothing is reserved yet.
2. `{"session_id": "...", "message": "Add one medu vada."}` → a new proposal
   with the new total (still nothing reserved).
3. `{"session_id": "...", "message": "Yes", "confirm_proposal_id": "<pending_order.proposal_id>"}`
   → `order` is `pending`, its stock is reserved, `order.payment.status` is `pending`.

Alternatively place an order directly: `POST /api/v1/restaurants/{id}/orders` with
`{"customer_phone": "+91 98765 43210", "items": [{"menu_item_id": "...", "quantity": 2}]}`.

### 4. Simulate the payment (staff)

`POST /api/v1/payments/{order.payment.id}/simulate` with
`{"outcome": "succeeded", "amount_paise": <order.total_paise>}` → the order becomes
`confirmed` and its stock moves from reserved to sold. Use `"outcome": "failed"`
to see the order cancelled and the stock released, or wait past
`ORDER_RESERVATION_TTL_SECONDS` (or set a past `reservation_expires_at` in a dev
database and call `POST /api/v1/maintenance/expire-reservations`) to see an
expiry. Ask the agent "What is my payment status?" at any point: it reads the
stored status.

### 5. Fulfil and inspect

`PATCH /api/v1/orders/{id}/status` with `preparing`, then `ready`, then `completed`.
Inspect:

- `GET /api/v1/orders/{id}`: lines, totals, `inventory_state`, payment;
- `GET /api/v1/restaurants/{id}/inventory`: on hand / reserved / available / sold;
- `GET /api/v1/menu-items/{item_id}/inventory/movements`: the stock ledger
  (`restock`, `reserve`, `sell`, …);
- `GET /api/v1/orders/{id}/track` with the order's `X-Order-Token`: the customer's view.

### 6. By phone (Exotel)

Once Exotel's Voicebot applet points at this server (see *Phone calls* above), the
same flow works by voice: the caller orders, changes the cart, says "yes" or
presses 1 to place it, hears that the demo payment is pending, and can ask for
the status after staff simulate the payment.

---

## Running Tests

### 1. Automated test suite

```powershell
python -m pytest -q                  # everything
python -m pytest -q -m "not live_db" # unit, mocked and migration tests only
python -m pytest -q -m live_db       # real-PostgreSQL tests only
```

Tests marked `live_db` (`test_*_live.py`) run against the configured database,
inside a transaction that is always rolled back, so they never leave data behind.
Row counts are scoped to test restaurants (slugs starting `test-`), so existing demo
data does not affect results. The exception to rollback is
`test_orders_concurrency_live.py`: real concurrent transactions need committed data,
so it commits a throwaway restaurant and always deletes it (with its orders,
payments, stock and ledger rows) afterwards, verifying nothing is left. It covers
customers racing for limited stock, duplicate payment callbacks and payment racing
the expiry sweep. Live tests are skipped
automatically when PostgreSQL is unreachable or not migrated to head. All other
tests need no database.

### 2. Standalone live database verification

If PostgreSQL 18.6 is running and `.env` has been configured with valid credentials:

```powershell
python tests/live_db_check.py
```

This verifies:
1. SQLAlchemy engine connection to PostgreSQL via Psycopg 3.
2. `SELECT 1` execution and server version detection.
3. Connected user and database verification.
4. Alembic configuration loading and script location.

---

## Project Structure

```
restaurant_voice_ai_agent/
├── alembic/
│   ├── env.py            # Alembic runtime environment (reads app settings)
│   ├── script.py.mako    # Migration script template
│   └── versions/         # Migration revision scripts
├── app/
│   ├── __init__.py       # Package marker
│   ├── config.py         # Pydantic Settings — loads from .env and environment
│   ├── database.py       # Engine, sessionmaker, Base, and get_db dependency
│   ├── main.py           # FastAPI app, exception handlers, routers, health, expiry sweep
│   ├── security.py       # Staff key and per-order customer tokens
│   ├── demo_seed.py      # python -m app.demo_seed: demo restaurant, menu and stock
│   ├── models/           # Restaurant, MenuItem, Order(Item), InventoryItem/Movement, Payment
│   ├── schemas/          # Pydantic request/response schemas
│   ├── services/         # Database operations and domain errors
│   ├── agent/            # LLM provider, tools, sessions, prompt, agent loop
│   ├── telephony/        # Exotel Voicebot WebSocket, speech provider, audio/VAD
│   └── routers/          # HTTP endpoints (thin; delegate to services/agent)
├── tests/
│   ├── conftest.py       # Live-DB fixtures (rolled-back transactions)
│   ├── test_health.py    # /health endpoint tests
│   ├── test_database.py  # Database config, session cleanup, and /health/db tests
│   ├── test_models.py    # Model metadata unit tests
│   ├── test_migrations.py # Revision chain and offline migration SQL
│   ├── test_schemas.py   # Request validation unit tests
│   ├── test_api_mocked.py # API tests with a mocked session
│   ├── test_models_live.py # Model behaviour against real PostgreSQL
│   ├── test_api_live.py  # API CRUD flows against real PostgreSQL
│   ├── test_orders_unit.py # Order metadata, lifecycle, pricing, schemas
│   ├── test_orders_api_mocked.py # Order API mapping and failure paths (mocked)
│   ├── test_orders_live.py # Order API flows against real PostgreSQL
│   ├── test_orders_concurrency_live.py # Concurrent ordering (committed, cleaned up)
│   ├── agent_fakes.py    # Scripted LLM provider for tests (no network)
│   ├── test_agent_unit.py # Agent schemas, tools, guard, sessions, provider (HTTP mocked)
│   ├── test_agent_api_mocked.py # Agent endpoint with mocked DB and scripted LLM
│   ├── test_agent_live.py # Agent tools and confirmation against real PostgreSQL
│   ├── test_telephony_exotel.py # Audio/VAD, speech provider, Exotel call WebSocket (faked)
│   ├── test_inventory_payments_live.py # Stock, reservations, demo payments, expiry, auth, agent
│   ├── test_demo_seed_live.py # Seed is idempotent and keeps live stock
│   ├── db_helpers.py     # Row counts scoped to test data
│   └── live_db_check.py  # Standalone live PostgreSQL verification script
├── docs/                 # Architecture notes and documentation
├── .env.example          # Safe environment variable template (committed)
├── .gitignore            # Git exclusion rules (ignores .env, .venv, etc.)
├── alembic.ini           # Alembic configuration (credentials not hardcoded)
├── pyproject.toml        # Authoritative project metadata and dependencies
└── README.md
```

---

## Environment Variables Reference

See [`.env.example`](.env.example) for the full list with descriptions.

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `Restaurant AI Voice Agent` | Application display name |
| `APP_ENV` | `development` | Runtime environment (`development`, `staging`, `production`) |
| `DEBUG` | `true` | Enables `/docs` and `/redoc` |
| `DATABASE_URL` | `None` | Psycopg 3 connection URL: `postgresql+psycopg://user:pass@host:port/dbname` |
| `DB_POOL_SIZE` | `5` | SQLAlchemy connection pool size |
| `DB_POOL_MAX_OVERFLOW` | `10` | Max overflow connections above pool size |
| `DB_ECHO_SQL` | `false` | Log raw SQL statements (keep `false` in production) |
| `DB_CONNECT_TIMEOUT` | `5` | Seconds to wait for a PostgreSQL connection before failing |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `LLM_API_KEY` | `None` | Provider API key (keep in `.env` only) |
| `LLM_MODEL` | `None` | Model name; must support tool calling |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-request provider timeout |
| `LLM_MAX_TOKENS` | `500` | Max tokens per model reply |
| `LLM_TEMPERATURE` | unset | Sent only if set |
| `AGENT_MAX_MODEL_CALLS` | `4` | Model round-trips per chat request |
| `AGENT_MAX_TOOL_CALLS` | `6` | Tool executions per chat request |
| `AGENT_SESSION_TTL_SECONDS` | `1800` | Idle time before a session expires |
| `AGENT_MAX_SESSIONS` | `1000` | Sessions kept in memory (least recently used evicted) |
| `LLM_RATE_LIMIT_MAX_WAIT_SECONDS` | `15` | On HTTP 429, wait up to this `Retry-After` once and retry |
| `SPEECH_RATE_LIMIT_MAX_WAIT_SECONDS` | `3` | Same for speech requests |
| `LOG_FILE` | unset | Also write the app log to this file (rotated at 5 MB); e.g. `logs/server.log` |
| `STAFF_API_KEY` | `None` | `X-Staff-Key` for staff endpoints (unset: they answer 503) |
| `ORDER_TOKEN_SECRET` | `None` | HMAC secret for customer order tokens (unset: per-process, lost on restart) |
| `ORDER_RESERVATION_TTL_SECONDS` | `600` | How long an unpaid order holds its stock |
| `RESERVATION_SWEEP_SECONDS` | `60` | Expiry sweep interval (`0` disables the background sweep) |
| `DASHBOARD_SESSION_HOURS` | `12` | Dashboard sign-in lifetime |
| `DASHBOARD_LOW_STOCK_THRESHOLD` | `5` | "Low stock" level in the dashboard summary |
| `DASHBOARD_EVENTS_SECONDS` | `3` | How often the live-update stream checks for changes |
| `EXOTEL_*`, `SPEECH_*`, `STT_*`, `TTS_*`, `VAD_*` | see `.env.example` | Phone calls and speech |

---

## Known Limitations

- **Payments are simulated.** There is no payment gateway; nothing is charged or
  refunded. "Refunded" is a demo state only. Integrating a real gateway (webhook
  signature verification, reconciliation) is a separate piece of work.
- **Agent sessions live in process memory**: run one worker; carts in progress are
  lost on restart (placed orders are not: they are in PostgreSQL). The chat API has
  no caller authentication.
- **Customers cannot cancel by voice**; they cancel with their order token (or staff
  cancel). Order status *can* be asked by voice or chat.
- **No taxes, fees, discounts or delivery.**
- **Groq free tier**: about 8,000 LLM tokens per minute, roughly two agent turns per
  minute. Short waits are retried; longer ones make the assistant answer "temporarily
  unavailable" (by phone, the call apologises and ends). Real calls need a paid tier
  or a smaller prompt/model.
- **English only** on the phone (see *Language limitation* above).
- Security has been tested at the level described here (staff key, order tokens,
  no secrets in logs or errors); it has not been audited for production.
