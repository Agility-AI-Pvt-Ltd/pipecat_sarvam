"""Check the voice agent layer by layer, without a phone or Vobiz.

    python scripts/smoke_test.py                       # 1. LLM key   2. Sarvam TTS+STT
    python scripts/smoke_test.py --call                # 3. a full call against a running agent
    python scripts/smoke_test.py --call --url ws://<host>:8100/vobiz/ws

Layer 3 plays a synthesised caller into the agent's /vobiz/ws exactly as Vobiz
would — 16 kHz L16 in 20 ms frames, in real time — and records the bot's reply.
It reports what the agent heard, what it said back, and how long the caller waited
in silence before the first word. That last number is the one a customer feels.

No call_id is sent, so the agent never reports a test call to the CRM.
Reads .env from the working directory, like the agent does. Prints no secrets.
"""

from __future__ import annotations

import argparse
import asyncio
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop  # removed in 3.13; this project pins 3.11
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

from dotenv import dotenv_values

RATE = 16000
FRAME_BYTES = RATE * 2 // 50  # 20 ms of 16-bit mono
CALLER_LINE = "Hello, mujhe Noida mein 3BHK flat chahiye, budget ek crore hai."
OUT = Path(__file__).resolve().parent / "out"


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def _bad(msg: str) -> None:
    print(f"  FAIL  {msg}")


def _warn(msg: str) -> None:
    print(f"  WARN  {msg}")


# --------------------------------------------------------------------------- 1. LLM
def check_llm(env: dict) -> bool:
    print("\n[1] LLM — OPENAI_* (OpenRouter by default)")
    key, base, model = env.get("OPENAI_API_KEY"), env.get("OPENAI_BASE_URL"), env.get("OPENAI_MODEL")
    if not key:
        _bad("OPENAI_API_KEY is empty")
        return False
    base = (base or "https://openrouter.ai/api/v1").strip().rstrip("/")
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a real-estate assistant on a phone call. Reply in Hinglish, "
                "Roman script only, one or two short sentences, no markdown.",
            },
            {"role": "user", "content": CALLER_LINE},
        ],
        "max_tokens": 180,
        "temperature": 0.4,
    }
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        _bad(f"HTTP {exc.code} from {base}: {detail}")
        if exc.code == 401:
            print("        The key was refused. With OpenRouter it should start sk-or-v1-.")
        if exc.code == 429:
            print("        Rate-limited. openrouter/free allows 20 requests/min and 50/day.")
        return False
    except Exception as exc:  # noqa: BLE001 - a smoke test reports, it does not crash
        _bad(f"could not reach {base}: {type(exc).__name__}: {exc}")
        return False

    elapsed = time.time() - started
    reply = (payload["choices"][0]["message"].get("content") or "").strip()
    usage = payload.get("usage") or {}
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    if not reply:
        _bad(f"empty reply from {payload.get('model')} ({elapsed:.1f}s)")
        return False
    _ok(f"{elapsed:.1f}s, routed to {payload.get('model')}")
    print(f"        reply: {reply[:200]}")
    if elapsed > 2.5:
        _warn(f"{elapsed:.1f}s is long for a phone turn; the caller hears this as silence")
    if reasoning:
        _warn(f"model spent {reasoning} reasoning tokens before answering — dead air on a call")
    return True


# ------------------------------------------------------------------------ 2. Sarvam
def _sarvam(env: dict):
    from sarvamai import SarvamAI

    return SarvamAI(api_subscription_key=env["SARVAM_API_KEY"])


def synthesise(env: dict, text: str) -> bytes:
    """Sarvam TTS to raw 16 kHz mono 16-bit PCM."""
    response = _sarvam(env).text_to_speech.convert(
        text=text,
        target_language_code=env.get("SARVAM_TTS_LANGUAGE") or "hi-IN",
        speaker=env.get("SARVAM_TTS_VOICE") or "shubh",
        model=env.get("SARVAM_TTS_MODEL") or "bulbul:v3",
        speech_sample_rate=RATE,
    )
    with wave.open(io.BytesIO(base64.b64decode(response.audios[0]))) as wav:
        pcm = wav.readframes(wav.getnframes())
        if wav.getnchannels() == 2:
            pcm = audioop.tomono(pcm, wav.getsampwidth(), 0.5, 0.5)
        if wav.getsampwidth() != 2:
            pcm = audioop.lin2lin(pcm, wav.getsampwidth(), 2)
        if wav.getframerate() != RATE:
            pcm, _ = audioop.ratecv(pcm, 2, 1, wav.getframerate(), RATE, None)
    return pcm


def transcribe(env: dict, pcm: bytes) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm)
    buffer.seek(0)
    buffer.name = "audio.wav"
    response = _sarvam(env).speech_to_text.transcribe(
        file=buffer, model="saaras:v3", mode=env.get("SARVAM_MODE") or "transcribe"
    )
    return (response.transcript or "").strip()


def check_sarvam(env: dict) -> bytes | None:
    print("\n[2] Sarvam — TTS then STT round trip")
    if not env.get("SARVAM_API_KEY"):
        _bad("SARVAM_API_KEY is empty")
        return None
    try:
        started = time.time()
        pcm = synthesise(env, CALLER_LINE)
        _ok(f"TTS {len(pcm) / (RATE * 2):.1f}s of audio in {time.time() - started:.1f}s")
    except Exception as exc:  # noqa: BLE001
        _bad(f"TTS failed: {type(exc).__name__}: {str(exc)[:300]}")
        return None
    try:
        started = time.time()
        heard = transcribe(env, pcm)
        _ok(f"STT in {time.time() - started:.1f}s")
        print(f"        said : {CALLER_LINE}")
        print(f"        heard: {heard}")
    except Exception as exc:  # noqa: BLE001
        _bad(f"STT failed: {type(exc).__name__}: {str(exc)[:300]}")
    return pcm


# ------------------------------------------------------------------- 3. full call
async def check_call(env: dict, url: str, caller_pcm: bytes) -> bool:
    print(f"\n[3] Full call against {url}")
    import websockets

    stream_id = f"smoke-{uuid.uuid4().hex[:8]}"
    heard: list[str] = []
    bot_audio = bytearray()
    first_audio_at: float | None = None
    last_audio_at: float | None = None
    caller_done_at: float | None = None
    interruptions = 0

    try:
        socket = await websockets.connect(url, max_size=None, open_timeout=10)
    except Exception as exc:  # noqa: BLE001
        _bad(f"could not connect: {type(exc).__name__}: {exc}")
        print("        Is the agent running? Local: uvicorn pipecat_sarvam_vobiz.main:app --port 8100")
        return False

    async def receive() -> None:
        nonlocal first_audio_at, last_audio_at, interruptions
        async for raw in socket:
            message = json.loads(raw)
            event = message.get("event")
            if event == "playAudio":
                now = time.time()
                first_audio_at = first_audio_at or now
                last_audio_at = now
                bot_audio.extend(base64.b64decode(message["media"]["payload"]))
            elif event == "clearAudio":
                interruptions += 1
            elif event == "sarvamSTT" and (message.get("stt") or {}).get("kind") == "final":
                heard.append(message["stt"]["text"])

    async def send(pcm: bytes) -> None:
        for offset in range(0, len(pcm), FRAME_BYTES):
            frame = pcm[offset : offset + FRAME_BYTES].ljust(FRAME_BYTES, b"\0")
            await socket.send(
                json.dumps(
                    {
                        "event": "media",
                        "streamId": stream_id,
                        "media": {"payload": base64.b64encode(frame).decode()},
                    }
                )
            )
            await asyncio.sleep(0.02)  # real time, as Vobiz streams it

    receiver = asyncio.create_task(receive())
    try:
        await socket.send(
            json.dumps(
                {
                    "event": "start",
                    "sequenceNumber": 0,
                    "start": {
                        "callId": stream_id,
                        "streamId": stream_id,
                        # Makes the agent forward its STT transcripts to us.
                        "accountId": "browser-test",
                        "tracks": ["inbound"],
                        "mediaFormat": {"encoding": "audio/x-l16", "sampleRate": RATE},
                    },
                }
            )
        )
        silence = b"\0" * FRAME_BYTES * 50  # 1 s
        await send(silence)  # let the Sarvam sockets connect
        await send(caller_pcm)
        caller_done_at = time.time()
        # A phone line never goes quiet; keep streaming silence while waiting.
        deadline = caller_done_at + 25
        while time.time() < deadline:
            await send(b"\0" * FRAME_BYTES * 10)
            if last_audio_at and time.time() - last_audio_at > 2.0:
                break
    finally:
        receiver.cancel()
        await socket.close()

    print(f"        agent heard: {' '.join(heard) or '(nothing forwarded)'}")
    if not bot_audio:
        _bad("no audio came back within 25 s")
        print("        Check the agent's terminal — the first error there names the failing layer.")
        return False

    wait = first_audio_at - caller_done_at
    seconds = len(bot_audio) / (RATE * 2)
    OUT.mkdir(exist_ok=True)
    path = OUT / "bot_reply.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(bytes(bot_audio))
    _ok(f"bot replied: {seconds:.1f}s of audio, saved to {path.relative_to(Path.cwd())}")
    print(f"        caller waited {wait:.2f}s in silence before the first word")
    if wait > 2.0:
        _warn("over 2 s feels broken on a phone call — usually the LLM; see layer 1's timing")
    if interruptions:
        _warn(f"{interruptions} clearAudio event(s) — the bot was interrupted (barge-in)")
    try:
        print(f"        bot said   : {transcribe(env, bytes(bot_audio))}")
    except Exception as exc:  # noqa: BLE001
        _warn(f"could not transcribe the reply: {exc}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--call", action="store_true", help="also run a full call against a running agent")
    parser.add_argument("--url", default="ws://127.0.0.1:8100/vobiz/ws", help="agent WebSocket URL")
    args = parser.parse_args()

    env = {k: (v or "").strip() for k, v in dotenv_values(".env").items()}
    if not env:
        print("No .env in the current directory — run this from the repo root.")
        return 2

    llm_ok = check_llm(env)
    caller_pcm = check_sarvam(env)
    call_ok = True
    if args.call:
        if caller_pcm is None:
            print("\n[3] skipped — needs Sarvam TTS to synthesise the caller")
            call_ok = False
        else:
            call_ok = asyncio.run(check_call(env, args.url, caller_pcm))

    passed = llm_ok and caller_pcm is not None and call_ok
    print("\nRESULT:", "all layers passed" if passed else "something failed — see above")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
