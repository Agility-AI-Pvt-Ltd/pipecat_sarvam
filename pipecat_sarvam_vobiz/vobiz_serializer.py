from __future__ import annotations

import base64
import json
from typing import Any

from loguru import logger
from pydantic import Field

from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class VobizFrameSerializer(FrameSerializer):
    """Translate Vobiz Stream JSON events to Pipecat audio frames and back."""

    class InputParams(FrameSerializer.InputParams):
        stream_sample_rate: int = 16000
        stream_encoding: str = "audio/x-l16"
        sample_rate: int | None = None
        include_stream_id_on_play_audio: bool = True
        resampler_clear_after_secs: float = Field(default=10.0, ge=0.0)

    def __init__(self, params: InputParams | None = None):
        params = params or VobizFrameSerializer.InputParams()
        super().__init__(params=params)
        self._params: VobizFrameSerializer.InputParams = params
        self._stream_id: str | None = None
        self._call_id: str | None = None
        self._vobiz_sample_rate = params.stream_sample_rate
        self._vobiz_encoding = params.stream_encoding
        self._sample_rate = params.sample_rate or params.stream_sample_rate
        self._input_resampler = create_stream_resampler(
            clear_after_secs=params.resampler_clear_after_secs
        )
        self._output_resampler = create_stream_resampler(
            clear_after_secs=params.resampler_clear_after_secs
        )

    async def setup(self, frame: StartFrame):
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    @property
    def stream_id(self) -> str | None:
        return self._stream_id

    @property
    def call_id(self) -> str | None:
        return self._call_id

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, (EndFrame, CancelFrame)):
            if self._stream_id:
                return json.dumps({"event": "stop", "streamId": self._stream_id})
            return None

        if isinstance(frame, InterruptionFrame):
            if self._stream_id:
                return json.dumps({"event": "clearAudio", "streamId": self._stream_id})
            return None

        if isinstance(frame, AudioRawFrame):
            payload = await self._serialize_audio(frame)
            if not payload:
                return None
            answer: dict[str, Any] = {
                "event": "playAudio",
                "media": {
                    "contentType": self._vobiz_encoding,
                    "sampleRate": self._vobiz_sample_rate,
                    "payload": payload,
                },
            }
            if self._params.include_stream_id_on_play_audio and self._stream_id:
                answer["streamId"] = self._stream_id
            return json.dumps(answer)

        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Ignoring non-JSON Vobiz WebSocket message")
            return None

        event = message.get("event")
        if event == "start":
            start = message.get("start") or {}
            media_format = start.get("mediaFormat") or {}
            self._stream_id = start.get("streamId") or message.get("streamId")
            self._call_id = start.get("callId")
            self._vobiz_encoding = media_format.get("encoding") or self._vobiz_encoding
            self._vobiz_sample_rate = int(media_format.get("sampleRate") or self._vobiz_sample_rate)
            print(
                "[stream {stream}] started call={call} sample_rate={rate} encoding={encoding}".format(
                    stream=(self._stream_id or "unknown")[:8],
                    call=(self._call_id or "unknown"),
                    rate=self._vobiz_sample_rate,
                    encoding=self._vobiz_encoding,
                ),
                flush=True,
            )
            return None

        if event == "playedStream":
            logger.debug("Vobiz playback checkpoint reached: {}", message.get("name"))
            return None

        if event == "media":
            media = message.get("media") or {}
            payload_base64 = media.get("payload")
            if not payload_base64:
                return None
            payload = base64.b64decode(payload_base64)
            audio = await self._deserialize_audio(payload)
            if not audio:
                return None
            return InputAudioRawFrame(audio=audio, num_channels=1, sample_rate=self._sample_rate)

        logger.debug("Ignoring Vobiz event: {}", event)
        return None

    async def _deserialize_audio(self, payload: bytes) -> bytes:
        if self._vobiz_encoding == "audio/x-mulaw":
            return await ulaw_to_pcm(
                payload, self._vobiz_sample_rate, self._sample_rate, self._input_resampler
            )
        if self._vobiz_encoding == "audio/x-l16":
            return await self._input_resampler.resample(
                payload, self._vobiz_sample_rate, self._sample_rate
            )
        logger.warning("Unsupported Vobiz input audio encoding: {}", self._vobiz_encoding)
        return b""

    async def _serialize_audio(self, frame: AudioRawFrame) -> str:
        if self._vobiz_encoding == "audio/x-mulaw":
            audio = await pcm_to_ulaw(
                frame.audio, frame.sample_rate, self._vobiz_sample_rate, self._output_resampler
            )
        elif self._vobiz_encoding == "audio/x-l16":
            audio = await self._output_resampler.resample(
                frame.audio, frame.sample_rate, self._vobiz_sample_rate
            )
        else:
            logger.warning("Unsupported Vobiz output audio encoding: {}", self._vobiz_encoding)
            return ""
        return base64.b64encode(audio).decode("utf-8")
