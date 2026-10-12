from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import FileResponse
from loguru import logger

from pipecat_sarvam_vobiz.agent import run_vobiz_agent
from pipecat_sarvam_vobiz.call_session import CallRegistry, CallSession, CapacityError
from pipecat_sarvam_vobiz.settings import Settings, load_settings

load_dotenv(dotenv_path=Path.cwd() / ".env")
settings = load_settings()

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger.remove()
logger.add(lambda message: print(message, end=""), level=settings.log_level.upper())

#: One per process. This is the only shared mutable state in the service, and it
#: exists solely to bound how many calls this replica accepts. Everything else
#: about a call lives on its own WebSocket, which is what makes scaling out a
#: matter of running more of these.
registry = CallRegistry(
    max_concurrent=settings.max_concurrent_calls,
    connect_idle_seconds=settings.call_connect_idle_secs,
    media_idle_seconds=settings.call_media_idle_secs,
)


async def _reap_idle_sessions() -> None:
    """Close sockets that never started, or went silent after hangup."""
    while True:
        await asyncio.sleep(5)
        for token, closer in registry.pop_idle():
            print(f"[vobiz] idle-reap token={token}", flush=True)
            if closer is not None:
                try:
                    await closer()
                except Exception as exc:
                    logger.warning("Idle close failed for {}: {}", token, exc)
                    registry.release(token)
            else:
                registry.release(token)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(_reap_idle_sessions())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Pipecat Vobiz Voice Agent", lifespan=lifespan)
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _websocket_url(request: Request, settings: Settings, call_id: str | None) -> str:
    """Build the `wss://` URL Vobiz should stream to, carrying the call id.

    The call id arrives here as a query parameter on the answer URL — the CRM put
    it there when it asked Vobiz to dial — and is forwarded onto the WebSocket
    URL. That is the whole mechanism by which this service knows which CRM record
    a stream belongs to; Vobiz's own `callId` identifies the *phone call*, not the
    row it was created from.
    """
    if settings.vobiz_ws_url:
        base = settings.vobiz_ws_url
    elif settings.public_base_url:
        stripped = settings.public_base_url.rstrip("/")
        base = (
            f"{stripped.replace('https://', 'wss://').replace('http://', 'ws://')}/vobiz/ws"
        )
    else:
        url = str(request.url_for("vobiz_ws"))
        parsed = urlparse(url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        base = urlunparse(parsed._replace(scheme=scheme))

    if not call_id:
        return base
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}{urlencode({'call_id': call_id})}"


def _stream_xml(ws_url: str, content_type: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Stream bidirectional="true" keepCallAlive="true" contentType="{escape(content_type)}">'
        f"{escape(ws_url)}"
        "</Stream>"
        "</Response>"
    )


def _busy_xml() -> str:
    """What Vobiz gets when this replica is full.

    Hanging up immediately is the kind refusal. The alternative — accepting the
    stream anyway — degrades every conversation already running on this box, so a
    caller who hears nothing is better than twenty callers hearing stutter.
    """
    return '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> dict[str, int | bool]:
    """Live capacity, for a load balancer or a dashboard to read.

    `active_calls` against `capacity` is the number to watch when deciding how many
    replicas to run: if it sits at the ceiling, calls are being refused and the
    fleet needs another instance.
    """
    return registry.snapshot()


@app.get("/test")
async def test_client() -> FileResponse:
    return FileResponse(STATIC_DIR / "test_client.html")


@app.api_route("/vobiz/answer", methods=["GET", "POST"])
async def vobiz_answer(request: Request) -> Response:
    """Vobiz asks what to do with an answered call. Admission control lives here.

    This is the earliest point at which a call can be refused, and the last point
    at which refusing is cheap: after this the stream opens and a pipeline with
    live Sarvam STT/TTS sockets and an LLM client is built for it.
    """
    call_id = (request.query_params.get("call_id") or "").strip() or None
    try:
        # The reservation is keyed by call id, so the WebSocket that follows
        # converts this exact slot into a live call rather than taking a second
        # one. A call with no id (the browser test client, or a misconfigured
        # answer URL) cannot be correlated, so its reservation is cleared by the
        # TTL instead.
        registry.reserve(call_id)
    except CapacityError as exc:
        logger.warning("Refusing call {}: {}", call_id or "unknown", exc)
        return Response(content=_busy_xml(), media_type="application/xml")

    ws_url = _websocket_url(request, settings, call_id)
    print(f"[vobiz] answer call_id={call_id or 'none'} -> stream {ws_url}", flush=True)
    return Response(
        content=_stream_xml(ws_url, settings.vobiz_stream_content_type),
        media_type="application/xml",
    )


def _hangup_idents(request: Request, payload: bytes) -> list[str]:
    idents: list[str] = []
    hinted = (request.query_params.get("call_id") or "").strip()
    if hinted:
        idents.append(hinted)
    text = payload.decode("utf-8", errors="replace") if payload else ""
    if text:
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = {}
        if isinstance(body, dict):
            for key in ("call_id", "CallUUID", "RequestUUID", "callId", "request_uuid"):
                value = str(body.get(key) or "").strip()
                if value:
                    idents.append(value)
    return idents


@app.api_route("/vobiz/hangup", methods=["GET", "POST"])
async def vobiz_hangup(request: Request) -> dict[str, str]:
    """Vobiz hangup: drop the media socket so the capacity slot is freed.

    The CRM still owns the call record. This only closes a stream that would
    otherwise sit open on the public tunnel after the phone has already dropped.
    """
    payload = await request.body()
    printable = payload.decode("utf-8", errors="replace") if payload else ""
    print(f"[vobiz] hangup {printable}", flush=True)
    closer = registry.closer_for(*_hangup_idents(request, payload))
    if closer is not None:
        try:
            await closer()
        except Exception as exc:
            logger.warning("Hangup close failed: {}", exc)
    return {"status": "ok"}


@app.websocket("/vobiz/ws", name="vobiz_ws")
async def vobiz_ws(websocket: WebSocket) -> None:
    call_id = (websocket.query_params.get("call_id") or "").strip() or None
    session = CallSession(call_id=call_id, counts_toward_capacity=bool(call_id))
    token = call_id or f"probe-{id(websocket)}"

    async def disconnect() -> None:
        try:
            await websocket.close()
        except Exception:
            pass

    session.disconnect = disconnect
    session.on_provider_id = lambda provider_id: registry.bind_vobiz_id(token, provider_id)
    await websocket.accept()
    registry.activate(token, session)
    print(
        f"[vobiz] websocket connected call_id={call_id or 'none'} "
        f"active={registry.snapshot()['active_calls']} "
        f"counted={session.counts_toward_capacity}",
        flush=True,
    )
    try:
        await run_vobiz_agent(websocket, settings, session)
    finally:
        registry.release(token)
        print(
            f"[vobiz] websocket closed call_id={call_id or 'none'} "
            f"active={registry.snapshot()['active_calls']}",
            flush=True,
        )


def run() -> None:
    import uvicorn

    uvicorn.run(
        "pipecat_sarvam_vobiz.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
