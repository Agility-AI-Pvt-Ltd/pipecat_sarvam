# Pipecat + Sarvam + Vobiz Voice Agent

FastAPI service for a live Vobiz call stream. It answers calls with Vobiz XML,
opens a realtime WebSocket stream and runs each call through a Pipecat cascade:
Sarvam streaming speech-to-text, an LLM behind any OpenAI-compatible gateway
(OpenRouter by default), and Sarvam streaming text-to-speech back to the caller
through Vobiz `playAudio`. Turn-taking and barge-in are decided locally by Silero
VAD.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set:

```bash
SARVAM_API_KEY=...
OPENAI_API_KEY=sk-or-v1-...          # an OpenRouter key
OPENAI_MODEL=openrouter/free
OPENAI_BASE_URL=https://openrouter.ai/api/v1
PUBLIC_BASE_URL=https://your-ngrok-url.ngrok-free.app
```

Then run:

```bash
uvicorn pipecat_sarvam_vobiz.main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker compose up -d --build
docker compose logs -f voice-agent
```

The service listens on host port `8100` (container port 8000). CRM's API
already binds 8000 on the shared EC2. For that host, make sure the security
group allows inbound TCP `8100` from the callers/testers that need to
reach `/vobiz/answer` and `/vobiz/ws`. Point ngrok at `8100`, not 8000.

Configure Vobiz:

- Answer URL: `https://your-ngrok-url/vobiz/answer`
- Answer Method: `POST`
- Hangup URL: `https://your-ngrok-url/vobiz/hangup`
- Hangup Method: `POST`

When audio arrives, the terminal will show lines like:

```text
[stream c4dfd815] started call=5401fd2e sample_rate=16000 encoding=audio/x-l16
[partial] namaste mujhe...
[final] namaste mujhe kisan loan ke baare mein jaankari chahiye
```

The live pipeline is:

```text
Vobiz media ──► Silero VAD ──► Sarvam STT ──► LLM ──► Sarvam TTS ──► Vobiz playAudio
                 (local)       saaras:v3    OpenRouter   bulbul:v3
```

Everything runs at the Vobiz stream rate (16 kHz), which Sarvam STT, Silero and
Sarvam TTS all take natively, so nothing is resampled in the pipeline.

**Why Silero rather than Sarvam's VAD.** Sarvam's streaming STT can report speech
start and stop itself, but then every boundary costs a network round trip — on
the path where latency is audible, deciding the caller has finished and noticing
they have started talking over the bot. Silero runs on this box in milliseconds,
and when it reports silence Pipecat tells Sarvam to flush the transcript. Sarvam's
VAD is switched off so the two never disagree.

**Turn end** is a silence timeout on top of Silero (`USER_TURN_SILENCE_SECS`),
set explicitly. Pipecat 1.7's default is `LocalSmartTurnAnalyzerV3`, a second ML
model that would otherwise run on every call deciding when customers are done.

**Barge-in** is on by default, except during the bot's first reply, which can't
be interrupted — the first seconds of a phone call are the noisiest. The caller
can still speak first; the bot waits for them. `BARGE_IN=false` makes every reply
uninterruptible.

## Who speaks first

A CRM call (one with a `call_id`) opens with the `greeting` from the call context,
spoken the moment the stream starts (after `GREETING_DELAY_SECS`, default 0.6 s):
"Namaste Ravi ji! Noida Homes ki taraf se call hai…". Every organisation dials from
the same Vobiz number, so the business name has to be the first thing the customer
hears. The greeting is added to the LLM context, so the model continues from the
customer's answer instead of greeting again. The caller is muted until the greeting
finishes (`MuteUntilFirstBotCompleteUserMuteStrategy`), so a "Hello?" over it cannot
start a second reply. Without a greeting (browser test client, standalone use) the
caller speaks first, as before.

## Speaking the caller's language

Sarvam STT runs with language auto-detection, so each transcript is tagged with the
language the caller used. `CallerLanguageFollower` (in `language.py`) switches the
bulbul voice to match: immediately on the caller's first utterance, and after two
consecutive utterances in a new language from then on, so one stray word cannot flip
it. Hindi and English share the `hi-IN` voice, because the bot writes Hindi in Roman
script, which an `en-IN` voice would mispronounce. The LLM is told to answer in the
caller's language and script; it sees their words, so it needs no extra signal.

Supported: Hindi, English, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia,
Punjabi, Tamil, Telugu.

## Environment

| Variable | Default | Notes |
| --- | --- | --- |
| `SARVAM_API_KEY` | required | Sarvam key, used for both STT and TTS. |
| `SARVAM_MODE` | `transcribe` | saaras:v3 mode: `transcribe`, `translate`, `verbatim`, `translit`, `codemix`. |
| `SARVAM_TTS_MODEL` | `bulbul:v3` | Sarvam TTS model. |
| `SARVAM_TTS_VOICE` | `shubh` | Sarvam speaker. |
| `SARVAM_LANGUAGE` | auto | STT language. Unset/`auto` detects the caller's language per utterance; set e.g. `hi-IN` to pin it. |
| `SARVAM_TTS_LANGUAGE` | `hi-IN` | Voice language until the caller's language is detected; then it follows the caller (see below). |
| `SARVAM_TTS_PACE` | `1.05` | Speaking rate. |
| `SARVAM_TTS_MAX_CHUNK_LENGTH` | `120` | Characters per TTS chunk. |
| `OPENAI_API_KEY` | required | Key for the gateway in `OPENAI_BASE_URL` — an OpenRouter key by default. |
| `OPENAI_MODEL` | `openrouter/free` | Model id as the gateway names it. See the note below. |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible chat-completions endpoint. |
| `OPENAI_TEMPERATURE` | `0.4` | LLM temperature. |
| `OPENAI_MAX_COMPLETION_TOKENS` | `180` | Reply length cap, sent as `max_tokens`. A latency control on a call. |
| `SYSTEM_PROMPT` | built in | Concise Hinglish in Roman script, no markdown. Per-campaign scripts from the CRM override it. |
| `VAD_CONFIDENCE` | `0.7` | Silero speech confidence threshold. |
| `VAD_START_SECS` | `0.2` | Speech needed before a turn starts. |
| `VAD_STOP_SECS` | `0.2` | Silence before Silero reports the caller stopped. |
| `VAD_MIN_VOLUME` | `0.6` | Raise if line noise keeps interrupting the bot. |
| `USER_TURN_SILENCE_SECS` | `0.6` | Pause allowed before the turn ends. Bot answers after ≈ `VAD_STOP_SECS` + this. |
| `BARGE_IN` | `true` | Caller can interrupt the bot (never its first reply). `false` = never. |
| `PUBLIC_BASE_URL` | derived from request | Public HTTPS base URL used to build the Vobiz WebSocket URL. Use this with ngrok/cloudflared. |
| `VOBIZ_WS_URL` | derived from `PUBLIC_BASE_URL` | Full `wss://.../vobiz/ws` override. |
| `VOBIZ_STREAM_CONTENT_TYPE` | `audio/x-l16;rate=16000` | Vobiz stream format. Also accepts `audio/x-mulaw;rate=8000`. |
| `LOG_LEVEL` | `INFO` | Python logging level. |
| `BCRM_BASE_URL` | unset | CRM base URL for outcome reporting. Unset disables reporting. |
| `INTERNAL_API_TOKEN` | unset | Shared secret for the CRM's `/api/v1/internal/*` endpoints. |
| `MAX_CONCURRENT_CALLS` | `10` | Calls this process accepts before refusing. Measure it; see Concurrency. |
| `CALL_SUMMARY_ENABLED` | `true` | Summarise the transcript after the call ends. |
| `CALL_SUMMARY_MODEL` | `OPENAI_MODEL` | Model for the post-call summary, on the same gateway. |

**About `openrouter/free`.** It routes each request to a randomly chosen free
model, and free models are limited to **20 requests/minute and 50/day** (1,000/day
once $10 of credits have been bought). Every reply in a call is one request, so it
is fine for trying the agent and not for running a campaign: at 20/minute the
whole process can sustain about two live calls, and 50/day is two or three calls.
A rate-limited reply is silence on the line. Because the model changes from turn to
turn, the persona and the Hinglish can also drift mid-call. For production, name
one fast, non-reasoning model from <https://openrouter.ai/models> — reasoning
models think before the first word, and on a phone call that is dead air.

## Routes

- `GET /health` - service health.
- `GET /metrics` - live capacity: `active_calls`, `reserved_slots`, `capacity`, `accepting`.
- `GET /test` - browser WebSocket test client for microphone streaming.
- `POST /vobiz/answer` - returns Vobiz XML with realtime bidirectional `<Stream>`.
- `POST /vobiz/hangup` - logs Vobiz hangup webhook payload.
- `WS /vobiz/ws` - Vobiz realtime audio stream endpoint.

## Where this service sits

This is one of two applications. It owns the **live call** and nothing else:

```text
CRM (bcrm-backend)          this service
├── decides who to call     ├── runs the conversation
├── Temporal orchestration  ├── one Pipecat pipeline per call
├── retries and scheduling  └── reports the outcome back
└── owns all business state
```

The CRM asks Vobiz to dial, pointing the `answer_url` at this service with a
`?call_id=` attached. Vobiz fetches that URL when the customer picks up, this
service returns the `<Stream>` XML with the same `call_id` forwarded onto the
WebSocket URL, and from then on the media flows here while **call status
callbacks go to the CRM instead**. Splitting them that way keeps a business API
out of the audio path.

When the call ends, this service posts the transcript and a summarised outcome to
`POST {BCRM_BASE_URL}/api/v1/internal/calls/{call_id}/outcome`, authenticated
with `INTERNAL_API_TOKEN`. It never talks to Temporal. Its entire vocabulary is
"call X was answered" and "call X ended, here is what happened", which is what
lets it be pointed at a different product without changes.

Running standalone still works exactly as before: with `BCRM_BASE_URL` unset,
nothing is reported and the call is served normally.

## Concurrency

Each call gets its own `CallSession`, its own Pipecat pipeline, its own Sarvam
STT and TTS sockets and its own Silero instance. No call state lives in a module-level variable, so
scaling out is just running more replicas behind a load balancer:

```text
voice-agent-1   voice-agent-2   voice-agent-3
 MAX_CONCURRENT  MAX_CONCURRENT  MAX_CONCURRENT
```

`MAX_CONCURRENT_CALLS` bounds how many this process accepts. Past that,
`/vobiz/answer` returns `<Hangup/>` rather than accepting a call it cannot serve —
refusing the 21st call cleanly is better than degrading twenty live conversations.

**Do not trust the default.** The real ceiling is whichever of these is lowest on
your hardware and plan, and only load testing finds it:

```text
Pipecat CPU/memory per pipeline (Silero runs on CPU, per call)
LLM gateway rate limit — 20 requests/minute on openrouter/free
Sarvam STT/TTS concurrent-stream limits on your plan
Vobiz calls-per-second and concurrent-call limits
```

Watch `GET /metrics` during a load test. If `active_calls` sits at `capacity`,
calls are being refused and the fleet needs another replica.

## Browser Test Client

With the FastAPI server running, open:

```text
http://127.0.0.1:8000/test
```

Open the page through FastAPI, not as a local `file://.../test_client.html`
file. If opened from disk, browser URL detection may not know the server host.

Click `Start` to send a Vobiz-like `start` event and continuous 20 ms
`audio/x-l16` microphone frames to `/vobiz/ws`. The page does not use
client-side VAD; it streams until you click `Stop`.
