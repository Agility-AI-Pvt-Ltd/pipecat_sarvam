from __future__ import annotations

import os
from dataclasses import dataclass


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    sarvam_api_key: str | None
    openai_api_key: str | None
    openai_model: str
    system_prompt: str
    public_base_url: str | None
    vobiz_ws_url: str | None
    vobiz_stream_content_type: str
    sarvam_mode: str
    sarvam_language: str | None
    sarvam_vad_signals: bool
    sarvam_tts_model: str
    sarvam_tts_voice: str
    sarvam_tts_language: str
    sarvam_tts_pace: float
    sarvam_tts_min_buffer_size: int
    sarvam_tts_max_chunk_length: int
    log_level: str

    @property
    def stream_sample_rate(self) -> int:
        marker = "rate="
        if marker not in self.vobiz_stream_content_type:
            return 16000
        return int(self.vobiz_stream_content_type.split(marker, 1)[1].split(";", 1)[0])

    @property
    def stream_encoding(self) -> str:
        return self.vobiz_stream_content_type.split(";", 1)[0]


def load_settings() -> Settings:
    return Settings(
        sarvam_api_key=os.getenv("SARVAM_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            (
                "You are a helpful, warm, and concise voice assistant speaking on a phone call. "
                "Answer the caller directly, ask one clear follow-up question when needed, and keep "
                "responses short enough to sound natural when spoken. Prefer Hinglish in Roman script "
                "for Hindi/English callers, for example: 'Haan, ek bottle paani de dunga.' Do not use "
                "Devanagari unless the caller specifically asks for Hindi script. Match other caller "
                "languages naturally when needed, including Indian-language code-mixing."
            ),
        ),
        public_base_url=os.getenv("PUBLIC_BASE_URL"),
        vobiz_ws_url=os.getenv("VOBIZ_WS_URL"),
        vobiz_stream_content_type=os.getenv(
            "VOBIZ_STREAM_CONTENT_TYPE", "audio/x-l16;rate=16000"
        ),
        sarvam_mode=os.getenv("SARVAM_MODE", "transcribe"),
        sarvam_language=os.getenv("SARVAM_LANGUAGE"),
        sarvam_vad_signals=_bool_env("SARVAM_VAD_SIGNALS", True),
        sarvam_tts_model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"),
        sarvam_tts_voice=os.getenv("SARVAM_TTS_VOICE", "shubh"),
        sarvam_tts_language=os.getenv("SARVAM_TTS_LANGUAGE", "hi-IN"),
        sarvam_tts_pace=float(os.getenv("SARVAM_TTS_PACE", "1.05")),
        sarvam_tts_min_buffer_size=int(os.getenv("SARVAM_TTS_MIN_BUFFER_SIZE", "20")),
        sarvam_tts_max_chunk_length=int(os.getenv("SARVAM_TTS_MAX_CHUNK_LENGTH", "120")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
