# Pipecat + OpenAI Realtime + Vobiz Voice Agent

FastAPI service for a live Vobiz call stream. It answers calls with Vobiz XML,
opens a realtime WebSocket stream, sends caller audio into Pipecat, and uses
OpenAI Realtime for speech-to-speech reasoning, turn detection, barge-in, and
audio output back to the caller through Vobiz `playAudio`.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set:

```bash
OPENAI_API_KEY=...
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

The service listens on host port `8000`. For EC2, make sure the instance
security group allows inbound TCP `8000` from the callers/testers that need to
reach `/vobiz/answer` and `/vobiz/ws`.

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
Vobiz media -> OpenAI Realtime speech-to-speech -> Vobiz playAudio
```

OpenAI Realtime consumes audio directly. Input transcription is enabled for
debugging in the terminal and `/test` UI, but it is asynchronous guidance rather
than the exact internal representation used by the model.

## Environment

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | required | OpenAI API key for Realtime. |
| `OPENAI_REALTIME_MODEL` | `gpt-realtime-2.1` | OpenAI Realtime speech-to-speech model. |
| `OPENAI_REALTIME_VOICE` | `marin` | Realtime output voice. |
| `OPENAI_REALTIME_TRANSCRIPTION_MODEL` | `gpt-realtime-whisper` | Async input transcript model for logs and `/test` UI. |
| `OPENAI_REALTIME_NOISE_REDUCTION` | `near_field` | Realtime input noise reduction. Set empty to disable. |
| `OPENAI_REALTIME_VAD_THRESHOLD` | `0.65` | Server VAD activation threshold. Higher values require louder speech and can reduce noise triggers. |
| `OPENAI_REALTIME_VAD_PREFIX_PADDING_MS` | `500` | Audio included before detected speech. |
| `OPENAI_REALTIME_VAD_SILENCE_DURATION_MS` | `700` | Silence needed to close a turn. |
| `OPENAI_MAX_COMPLETION_TOKENS` | `180` | Max output tokens for each Realtime response. |
| `SYSTEM_PROMPT` | built in | General voice-agent prompt. Defaults to concise Hinglish in Roman script only. Override for your business/personality. |
| `PUBLIC_BASE_URL` | derived from request | Public HTTPS base URL used to build the Vobiz WebSocket URL. Use this with ngrok/cloudflared. |
| `VOBIZ_WS_URL` | derived from `PUBLIC_BASE_URL` | Full `wss://.../vobiz/ws` override. |
| `VOBIZ_STREAM_CONTENT_TYPE` | `audio/x-l16;rate=16000` | Vobiz stream format. Also accepts `audio/x-mulaw;rate=8000`. |
| `LOG_LEVEL` | `INFO` | Python logging level. |
| `BCRM_BASE_URL` | unset | CRM base URL for outcome reporting. Unset disables reporting. |
| `INTERNAL_API_TOKEN` | unset | Shared secret for the CRM's `/api/v1/internal/*` endpoints. |
| `MAX_CONCURRENT_CALLS` | `10` | Calls this process accepts before refusing. Measure it; see Concurrency. |
| `CALL_SUMMARY_ENABLED` | `true` | Summarise the transcript after the call ends. |
| `CALL_SUMMARY_MODEL` | `gpt-4o-mini` | Model used for the post-call summary. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Gateway for the summary call. |

Legacy Sarvam variables may still exist in older `.env` files, but they are not
used by the active OpenAI Realtime pipeline.

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

Each call gets its own `CallSession`, its own Pipecat pipeline and its own
OpenAI Realtime session. No call state lives in a module-level variable, so
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
Pipecat CPU/memory per pipeline
OpenAI Realtime rate + concurrency limits
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
# pipecat_sarvam
