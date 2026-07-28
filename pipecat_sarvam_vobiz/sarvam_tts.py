from __future__ import annotations

import json

from loguru import logger
from pipecat.services.sarvam.tts import SarvamTTSService


class SarvamV3TTSService(SarvamTTSService):
    """Sarvam streaming TTS with the current bulbul:v3 websocket config shape.

    Sarvam's live bulbul:v3 websocket currently rejects `min_buffer_size` with
    "Input parameters has to be a valid dictionary", even though the generated
    docs/SDK expose it. Keep this shim local so we can remove it when Pipecat or
    Sarvam updates the upstream contract.
    """

    async def _send_config(self):
        if not self._websocket:
            raise Exception("WebSocket not connected")

        config_data = {
            "target_language_code": self._settings.language,
            "speaker": self._settings.voice,
            "speech_sample_rate": int(self._speech_sample_rate),
            "enable_preprocessing": self._settings.enable_preprocessing,
            "max_chunk_length": self._settings.max_chunk_length,
            "output_audio_codec": self._output_audio_codec,
            "output_audio_bitrate": self._output_audio_bitrate,
            "pace": self._settings.pace,
            "model": self._settings.model,
        }
        if self._settings.pitch is not None:
            config_data["pitch"] = self._settings.pitch
        if self._settings.loudness is not None:
            config_data["loudness"] = self._settings.loudness
        if self._settings.temperature is not None:
            config_data["temperature"] = self._settings.temperature

        logger.debug(f"Config being sent is {config_data}")
        await self._websocket.send(json.dumps({"type": "config", "data": config_data}))
