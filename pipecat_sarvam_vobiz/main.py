from __future__ import annotations

import logging
from html import escape
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import FileResponse
from loguru import logger

from pipecat_sarvam_vobiz.agent import run_vobiz_agent
from pipecat_sarvam_vobiz.settings import Settings, load_settings

load_dotenv(dotenv_path=Path.cwd() / ".env")
settings = load_settings()

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger.remove()
logger.add(lambda message: print(message, end=""), level=settings.log_level.upper())

app = FastAPI(title="Pipecat Sarvam Vobiz Voice Agent")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _websocket_url(request: Request, settings: Settings) -> str:
    if settings.vobiz_ws_url:
        return settings.vobiz_ws_url

    if settings.public_base_url:
        base = settings.public_base_url.rstrip("/")
        return f"{base.replace('https://', 'wss://').replace('http://', 'ws://')}/vobiz/ws"

    url = str(request.url_for("vobiz_ws"))
    parsed = urlparse(url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse(parsed._replace(scheme=scheme))


def _stream_xml(ws_url: str, content_type: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Stream bidirectional="true" keepCallAlive="true" contentType="{escape(content_type)}">'
        f"{escape(ws_url)}"
        "</Stream>"
        "</Response>"
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/test")
async def test_client() -> FileResponse:
    return FileResponse(STATIC_DIR / "test_client.html")


@app.api_route("/vobiz/answer", methods=["GET", "POST"])
async def vobiz_answer(request: Request) -> Response:
    ws_url = _websocket_url(request, settings)
    xml = _stream_xml(ws_url, settings.vobiz_stream_content_type)
    print(f"[vobiz] answer -> stream {ws_url}", flush=True)
    return Response(content=xml, media_type="application/xml")


@app.api_route("/vobiz/hangup", methods=["GET", "POST"])
async def vobiz_hangup(request: Request) -> dict[str, str]:
    payload = await request.body()
    printable = payload.decode("utf-8", errors="replace") if payload else ""
    print(f"[vobiz] hangup {printable}", flush=True)
    return {"status": "ok"}


@app.websocket("/vobiz/ws", name="vobiz_ws")
async def vobiz_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    print("[vobiz] websocket connected", flush=True)
    await run_vobiz_agent(websocket, settings)
    print("[vobiz] websocket closed", flush=True)


def run() -> None:
    import uvicorn

    uvicorn.run(
        "pipecat_sarvam_vobiz.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
