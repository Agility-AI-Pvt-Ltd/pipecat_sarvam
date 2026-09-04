from __future__ import annotations

import asyncio
import os
import time

from fastapi import WebSocket

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai._constants import OPENAI_SAMPLE_RATE
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    AudioOutput,
    InputAudioNoiseReduction,
    InputAudioTranscription,
    PCMAudioFormat,
    Reasoning,
    SessionProperties,
    TurnDetection,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

try:
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
except ImportError:  # pragma: no cover - compatibility with older Pipecat releases.
    from pipecat.transports.network.fastapi_websocket import (  # type: ignore
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

from pipecat_sarvam_vobiz import outcome as outcome_module
from pipecat_sarvam_vobiz.bcrm_client import BcrmClient
from pipecat_sarvam_vobiz.call_session import CallSession
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


def build_openai_realtime(
    settings: Settings, *, system_prompt: str | None = None
) -> OpenAIRealtimeLLMService:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    noise_reduction = (
        InputAudioNoiseReduction(type=settings.openai_realtime_noise_reduction)
        if settings.openai_realtime_noise_reduction
        else None
    )
    session_properties = SessionProperties(
        output_modalities=["audio"],
        audio=AudioConfiguration(
            input=AudioInput(
                format=PCMAudioFormat(),
                transcription=InputAudioTranscription(
                    model=settings.openai_realtime_transcription_model,
                    language="hi",
                ),
                noise_reduction=noise_reduction,
                turn_detection=TurnDetection(
                    threshold=settings.openai_realtime_vad_threshold,
                    prefix_padding_ms=settings.openai_realtime_vad_prefix_padding_ms,
                    silence_duration_ms=settings.openai_realtime_vad_silence_duration_ms,
                ),
            ),
            output=AudioOutput(
                format=PCMAudioFormat(),
                voice=settings.openai_realtime_voice,
            ),
        ),
        reasoning=Reasoning(effort="low"),
        max_output_tokens=int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", "180")),
    )

    return OpenAIRealtimeLLMService(
        api_key=settings.openai_api_key,
        settings=OpenAIRealtimeLLMService.Settings(
            model=settings.openai_realtime_model,
            system_instruction=system_prompt or settings.system_prompt,
            session_properties=session_properties,
        ),
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

    # Fetched before the model is built, not on stream start, because the system
    # instruction is fixed when the Realtime session is created — after that the AI
    # is already listening and it is too late to tell it who it is.
    system_prompt = settings.system_prompt
    if session.call_id and bcrm.configured:
        call_context = await asyncio.to_thread(bcrm.fetch_context, session.call_id)
        if call_context:
            session.customer_name = call_context.get("customer_name") or None
            system_prompt = _compose_system_prompt(
                call_context.get("agent_prompt") or settings.system_prompt,
                customer_name=session.customer_name,
            )
            print(
                f"[bcrm] call {session.call_id} using "
                f"{'campaign' if call_context.get('agent_prompt') else 'default'} script",
                flush=True,
            )

    async def on_stream_start(info: dict) -> None:
        session.stream_id = info.get("stream_id")
        session.vobiz_call_id = info.get("call_id")
        session.answered_at = time.time()
        if session.call_id and bcrm.configured:
            await asyncio.to_thread(bcrm.report_answered, session.call_id)

    serializer = VobizFrameSerializer(
        VobizFrameSerializer.InputParams(
            stream_sample_rate=settings.stream_sample_rate,
            stream_encoding=settings.stream_encoding,
            sample_rate=OPENAI_SAMPLE_RATE,
        ),
        on_start=on_stream_start,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=OPENAI_SAMPLE_RATE,
            audio_out_sample_rate=OPENAI_SAMPLE_RATE,
            add_wav_header=False,
            serializer=serializer,
            session_timeout=None,
        ),
    )

    llm = build_openai_realtime(settings, system_prompt=system_prompt)
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)
    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            TerminalTranscriptLogger(source="openaiRealtime", session=session),
            llm,
            TerminalOpenAILogger(session=session),
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
    task = PipelineTask(
        pipeline,
        idle_timeout_secs=None,
        params=PipelineParams(
            audio_in_sample_rate=OPENAI_SAMPLE_RATE,
            audio_out_sample_rate=OPENAI_SAMPLE_RATE,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
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

    Both steps happen after the pipeline has stopped. Summarisation is a second
    LLM round-trip, and doing it while audio was still flowing would put it in
    contention with the Realtime session for the same rate limit.
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
        },
    )
