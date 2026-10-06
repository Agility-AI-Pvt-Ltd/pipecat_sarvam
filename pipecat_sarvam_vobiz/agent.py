"""One phone call: Vobiz audio in, Sarvam speech-to-text, an LLM, Sarvam speech out.

A cascade rather than a speech-to-speech model:

    Vobiz ──► Silero VAD ──► Sarvam STT (saaras:v3) ──► LLM ──► Sarvam TTS (bulbul:v3) ──► Vobiz
              (local)          (streaming)         (OpenRouter)      (streaming)

It replaced OpenAI Realtime, which needed an OpenAI key and did all three jobs in
one model. Splitting them costs a little latency and buys three things: Hindi
and Hinglish recognition from a model trained for it, a choice of LLM behind any
OpenAI-compatible gateway, and a transcript that is the STT's output rather than
a side channel of the speech model.

Everything runs at the Vobiz stream rate (16 kHz by default). Sarvam STT, Silero
and Sarvam TTS all take 16 kHz natively, so no audio is resampled in the pipeline.
"""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urlparse

from fastapi import WebSocket
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_mute import (
    AlwaysUserMuteStrategy,
    FirstSpeechUserMuteStrategy,
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

try:
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )
except ImportError:  # pragma: no cover - compatibility with older Pipecat releases.
    from pipecat.transports.network.fastapi_websocket import (  # type: ignore
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

from pipecat_sarvam_vobiz import outcome as outcome_module
from pipecat_sarvam_vobiz.bcrm_client import BcrmClient
from pipecat_sarvam_vobiz.call_session import CallSession
from pipecat_sarvam_vobiz.language import CallerLanguageFollower
from pipecat_sarvam_vobiz.sarvam_tts import SarvamV3TTSService
from pipecat_sarvam_vobiz.settings import Settings
from pipecat_sarvam_vobiz.transcript_logger import TerminalOpenAILogger, TerminalTranscriptLogger
from pipecat_sarvam_vobiz.vobiz_serializer import VobizFrameSerializer


def _compose_system_prompt(agent_prompt: str, *, customer_name: str | None) -> str:
    """The dealer's script, plus who is on the other end.

    The name is appended rather than substituted into the script, so a dealer who
    never mentions a placeholder still gets a call that greets the customer
    properly — and one whose CRM has no name for a number does not end up with the
    AI saying "Hello, {name}" out loud.
    """
    if not customer_name:
        return agent_prompt
    return f"{agent_prompt}\n\nThe person you are calling is named {customer_name}."


def stt_language(settings: Settings) -> Language | None:
    """`SARVAM_LANGUAGE`, or None for auto-detection (the default)."""
    value = (settings.sarvam_language or "").strip()
    if value.lower() in {"", "auto", "unknown"}:
        return None
    try:
        return Language(value)  # e.g. hi-IN
    except ValueError:
        return Language[value.upper().replace("-", "_")]  # e.g. HI_IN


def build_sarvam_stt(settings: Settings) -> SarvamSTTService:
    if not settings.sarvam_api_key:
        raise RuntimeError("SARVAM_API_KEY is required")

    return SarvamSTTService(
        api_key=settings.sarvam_api_key,
        sample_rate=settings.stream_sample_rate,
        keepalive_timeout=10.0,
        mode=settings.sarvam_mode,
        settings=SarvamSTTService.Settings(
            model="saaras:v3",
            # None = auto-detect: every transcript comes back tagged with the
            # language the caller used, and CallerLanguageFollower makes the
            # voice follow it. Set SARVAM_LANGUAGE (e.g. hi-IN) to pin one.
            language=stt_language(settings),
            # Off on purpose: Silero decides when the caller starts and stops.
            # With this false, Pipecat sends Sarvam a flush the moment Silero
            # reports silence, so the final transcript arrives without waiting
            # for Sarvam's own end-of-speech detection. Turning it on as well
            # would give two VADs a vote on every breath. See Settings.
            vad_signals=False,
        ),
    )


def _is_openrouter(base_url: str) -> bool:
    return (urlparse(base_url).hostname or "").endswith("openrouter.ai")


def build_llm(settings: Settings, *, system_prompt: str | None = None) -> OpenAILLMService:
    """The model that decides what to say. Any OpenAI-compatible endpoint.

    OpenRouter gets its own Pipecat service, which knows OpenRouter's quirks (it
    rewrites a second system message for Gemini models, which `openrouter/free`
    can route to). Anything else — vLLM, a local server, OpenAI itself — goes
    through the plain OpenAI service with `base_url` pointed at it.

    `max_tokens` rather than `max_completion_tokens`: the newer name is OpenAI's
    own and not every compatible server honours it, while `max_tokens` is the one
    parameter they all agree on. On a phone call the cap is not a cost control but
    a latency one — every extra sentence is the customer listening to TTS instead
    of answering.
    """
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required — with OPENAI_BASE_URL pointing at OpenRouter, "
            "this is an OpenRouter key (sk-or-v1-…), not an OpenAI one."
        )

    common = {
        "model": settings.openai_model,
        "system_instruction": system_prompt or settings.system_prompt,
        "temperature": settings.openai_temperature,
        "max_tokens": settings.openai_max_tokens,
    }
    if _is_openrouter(settings.openai_base_url):
        return OpenRouterLLMService(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            settings=OpenRouterLLMService.Settings(**common),
        )
    return OpenAILLMService(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        settings=OpenAILLMService.Settings(**common),
    )


def build_sarvam_tts(settings: Settings) -> SarvamTTSService:
    if not settings.sarvam_api_key:
        raise RuntimeError("SARVAM_API_KEY is required")

    return SarvamV3TTSService(
        api_key=settings.sarvam_api_key,
        sample_rate=settings.stream_sample_rate,
        settings=SarvamTTSService.Settings(
            model=settings.sarvam_tts_model,
            voice=settings.sarvam_tts_voice,
            language=settings.sarvam_tts_language,
            pace=settings.sarvam_tts_pace,
            min_buffer_size=settings.sarvam_tts_min_buffer_size,
            max_chunk_length=settings.sarvam_tts_max_chunk_length,
        ),
    )


def build_vad(settings: Settings) -> SileroVADAnalyzer:
    return SileroVADAnalyzer(
        sample_rate=settings.stream_sample_rate,
        params=VADParams(
            confidence=settings.vad_confidence,
            start_secs=settings.vad_start_secs,
            stop_secs=settings.vad_stop_secs,
            min_volume=settings.vad_min_volume,
        ),
    )


def build_user_mute_strategies(
    settings: Settings, *, greeting_first: bool = False
) -> list[BaseUserMuteStrategy]:
    """When the caller's audio is ignored.

    The bot's *first* speech is always protected. A phone line is noisiest in the
    first seconds — the connect click, the customer settling the handset — and
    letting that interrupt the opening line means the customer hears half a
    sentence and the bot starts again.

    Which strategy depends on who speaks first:

    * The bot greets first (`greeting_first`, every CRM call): mute from connect
      until the greeting has finished, so a "Hello?" spoken over it cannot start a
      second, overlapping reply.
    * The caller speaks first (no greeting — the browser test client, a standalone
      deployment): `FirstSpeechUserMuteStrategy`, which leaves the caller audible
      until the bot starts. Muting from connect here would swallow the very words
      the bot is waiting for, and the call would sit in silence.

    `BARGE_IN=false` mutes the caller for every bot reply — no interruptions at all.
    """
    if not settings.barge_in:
        return [AlwaysUserMuteStrategy()]
    if greeting_first:
        return [MuteUntilFirstBotCompleteUserMuteStrategy()]
    return [FirstSpeechUserMuteStrategy()]


def build_context_aggregator(
    settings: Settings, context: LLMContext, *, greeting_first: bool = False
) -> LLMContextAggregatorPair:
    return LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=build_vad(settings),
            # Explicit, because Pipecat's default stop strategy is
            # LocalSmartTurnAnalyzerV3: a second ML model, run on every call,
            # that would silently decide when customers are finished speaking.
            # A silence timeout on top of Silero is predictable and tunable.
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    SpeechTimeoutUserTurnStopStrategy(
                        user_speech_timeout=settings.user_turn_silence_secs
                    )
                ]
            ),
            user_mute_strategies=build_user_mute_strategies(
                settings, greeting_first=greeting_first
            ),
        ),
        assistant_params=LLMAssistantAggregatorParams(),
    )


async def run_vobiz_agent(
    websocket: WebSocket,
    settings: Settings,
    session: CallSession | None = None,
) -> None:
    """Run one call to completion, then report it.

    `session` carries the CRM's call id — taken from the answer URL and passed
    through the WebSocket URL — plus the transcript this call accumulates. It is
    created per connection, so nothing here is shared between concurrent calls.
    """
    session = session or CallSession()
    bcrm = BcrmClient(base_url=settings.bcrm_base_url, token=settings.internal_api_token)

    # Fetched before the LLM is built, because the system prompt is part of the
    # LLM's settings — it is who the bot is for the whole call.
    system_prompt = settings.system_prompt
    greeting: str | None = None
    if session.call_id and bcrm.configured:
        call_context = await asyncio.to_thread(bcrm.fetch_context, session.call_id)
        if call_context:
            session.customer_name = call_context.get("customer_name") or None
            # Spoken first, the moment the line opens. Every organisation calls
            # from one shared number, so the business name is the first thing the
            # customer must hear — not silence while the AI waits for "hello".
            greeting = (call_context.get("greeting") or "").strip() or None
            system_prompt = _compose_system_prompt(
                call_context.get("agent_prompt") or settings.system_prompt,
                customer_name=session.customer_name,
            )
            print(
                f"[bcrm] call {session.call_id} using "
                f"{'campaign' if call_context.get('agent_prompt') else 'default'} script",
                flush=True,
            )

    #: Filled once the task exists; the stream-start callback needs it to speak.
    running: dict[str, PipelineTask] = {}
    #: Holds the greeting task so it is not garbage-collected mid-flight.
    background: set[asyncio.Task] = set()

    async def speak_greeting() -> None:
        # A beat after pickup, so the first syllable is not lost while the phone
        # is still moving to the ear.
        await asyncio.sleep(settings.greeting_delay_secs)
        task = running.get("task")
        if task is None or not greeting:
            return
        session.record("assistant", greeting)
        # append_to_context: the LLM sees its own opening line, so it continues
        # the conversation instead of greeting a second time.
        await task.queue_frames([TTSSpeakFrame(greeting, append_to_context=True)])

    async def on_stream_start(info: dict) -> None:
        session.stream_id = info.get("stream_id")
        session.vobiz_call_id = info.get("call_id")
        session.answered_at = time.time()
        if greeting:
            pending = asyncio.create_task(speak_greeting())
            background.add(pending)
            pending.add_done_callback(background.discard)
        if session.call_id and bcrm.configured:
            await asyncio.to_thread(bcrm.report_answered, session.call_id)

    rate = settings.stream_sample_rate
    serializer = VobizFrameSerializer(
        VobizFrameSerializer.InputParams(
            stream_sample_rate=rate,
            stream_encoding=settings.stream_encoding,
            sample_rate=rate,
        ),
        on_start=on_stream_start,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=rate,
            audio_out_sample_rate=rate,
            add_wav_header=False,
            serializer=serializer,
            session_timeout=None,
        ),
    )

    stt = build_sarvam_stt(settings)
    llm = build_llm(settings, system_prompt=system_prompt)
    tts = build_sarvam_tts(settings)
    context = LLMContext()
    context_aggregator = build_context_aggregator(
        settings, context, greeting_first=bool(greeting)
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            CallerLanguageFollower(
                initial=settings.sarvam_tts_language, tts=tts, session=session
            ),
            TerminalTranscriptLogger(source="sarvam", session=session),
            context_aggregator.user(),
            llm,
            TerminalOpenAILogger(session=session),
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
    task = PipelineTask(
        pipeline,
        # No idle timeout: a customer thinking in silence is part of a call, not
        # a reason to end one. Vobiz ends the stream when the line drops.
        idle_timeout_secs=None,
        params=PipelineParams(
            audio_in_sample_rate=rate,
            audio_out_sample_rate=rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    running["task"] = task
    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    finally:
        # In a `finally` on purpose: a call that ends because the customer hung up
        # mid-sentence is the normal case, and its transcript is worth just as much
        # as one that ended cleanly.
        await report_call_outcome(settings, session)


async def report_call_outcome(settings: Settings, session: CallSession) -> None:
    """Summarise and hand the result to the CRM. Never raises.

    Both steps happen after the pipeline has stopped. Summarisation is one more
    LLM request on the same key as the live conversation, and on a rate-limited
    gateway (OpenRouter's free tier allows 20 a minute) a summary sent while other
    calls are mid-conversation can be the request that makes one of them go quiet.
    """
    if not session.call_id:
        print("[bcrm] no call_id on this stream; outcome not reported", flush=True)
        return

    client = BcrmClient(
        base_url=settings.bcrm_base_url, token=settings.internal_api_token
    )
    if not client.configured:
        return

    transcript = session.transcript_text()
    result = outcome_module.CallOutcome()
    if settings.call_summary_enabled and transcript:
        # urllib is blocking, and this runs on the event loop that other calls on
        # this replica are still using for their audio. Off to a thread it goes.
        result = await asyncio.to_thread(
            outcome_module.summarize,
            transcript,
            api_key=settings.openai_api_key,
            model=settings.call_summary_model,
            base_url=settings.openai_base_url,
        )

    await asyncio.to_thread(
        client.report_outcome,
        session.call_id,
        outcome=result.outcome,
        next_action=result.next_action,
        summary=result.summary,
        transcript=transcript or None,
        duration_seconds=session.duration_seconds(),
        metadata={
            "stream_id": session.stream_id,
            "vobiz_call_id": session.vobiz_call_id,
            "turns": len(session.turns),
            "language": session.language,
            # What the caller asked for, when the call collected it. The CRM
            # writes these onto the contact and its WhatsApp agent sends
            # matching homes; only keys the model filled in are present.
            "requirements": result.requirements,
        },
    )
