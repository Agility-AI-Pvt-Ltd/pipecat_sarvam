from __future__ import annotations

import os

from fastapi import WebSocket

from pipecat.audio.vad.silero import SileroVADAnalyzer
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
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transcriptions.language import Language

try:
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
except ImportError:  # pragma: no cover - compatibility with older Pipecat releases.
    from pipecat.transports.network.fastapi_websocket import (  # type: ignore
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

from pipecat_sarvam_vobiz.sarvam_tts import SarvamV3TTSService
from pipecat_sarvam_vobiz.settings import Settings
from pipecat_sarvam_vobiz.transcript_logger import TerminalOpenAILogger, TerminalTranscriptLogger
from pipecat_sarvam_vobiz.vobiz_serializer import VobizFrameSerializer


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
            language=Language.HI_IN,
            vad_signals=False,
        ),
    )


def build_openai_llm(settings: Settings) -> OpenAILLMService:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    return OpenAILLMService(
        api_key=settings.openai_api_key,
        settings=OpenAILLMService.Settings(
            model=settings.openai_model,
            system_instruction=settings.system_prompt,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.4")),
            max_completion_tokens=int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", "180")),
        ),
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


async def run_vobiz_agent(websocket: WebSocket, settings: Settings) -> None:
    serializer = VobizFrameSerializer(
        VobizFrameSerializer.InputParams(
            stream_sample_rate=settings.stream_sample_rate,
            stream_encoding=settings.stream_encoding,
            sample_rate=settings.stream_sample_rate,
        )
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=settings.stream_sample_rate,
            audio_out_sample_rate=settings.stream_sample_rate,
            add_wav_header=False,
            serializer=serializer,
            session_timeout=None,
        ),
    )

    stt = build_sarvam_stt(settings)
    llm = build_openai_llm(settings)
    tts = build_sarvam_tts(settings)
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_stop_timeout=0.5,
            vad_analyzer=SileroVADAnalyzer(sample_rate=settings.stream_sample_rate),
        ),
        assistant_params=LLMAssistantAggregatorParams(),
    )

    @stt.event_handler("on_speech_started")
    async def _on_speech_started(_service):
        print("[speech] started", flush=True)

    @stt.event_handler("on_speech_stopped")
    async def _on_speech_stopped(_service):
        print("[speech] stopped", flush=True)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            TerminalTranscriptLogger(),
            context_aggregator.user(),
            llm,
            TerminalOpenAILogger(),
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=settings.stream_sample_rate,
            audio_out_sample_rate=settings.stream_sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
