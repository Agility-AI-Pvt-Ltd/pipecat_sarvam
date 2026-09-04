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
    openai_realtime_model: str
    openai_realtime_voice: str
    openai_realtime_transcription_model: str
    openai_realtime_noise_reduction: str | None
    openai_realtime_vad_threshold: float
    openai_realtime_vad_prefix_padding_ms: int
    openai_realtime_vad_silence_duration_ms: int
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

    # --- CRM integration ----------------------------------------------------
    # Where to report call outcomes, and the shared secret to do it with. Both
    # unset is a valid configuration: the agent still answers calls, it just has
    # nowhere to send the transcript.
    bcrm_base_url: str | None
    internal_api_token: str | None
    #: Hard ceiling on simultaneous calls this process will accept. Determine it
    #: by load testing, not by guessing — see CallRegistry. Zero disables the
    #: ceiling, which is only sensible for local testing.
    max_concurrent_calls: int
    #: Post-call summarisation. Off when no OpenAI key is available; the CRM then
    #: receives the transcript without an interpretation of it.
    call_summary_enabled: bool
    call_summary_model: str
    openai_base_url: str

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
        openai_realtime_model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1"),
        openai_realtime_voice=os.getenv("OPENAI_REALTIME_VOICE", "marin"),
        openai_realtime_transcription_model=os.getenv(
            "OPENAI_REALTIME_TRANSCRIPTION_MODEL", "gpt-realtime-whisper"
        ),
        openai_realtime_noise_reduction=os.getenv(
            "OPENAI_REALTIME_NOISE_REDUCTION", "near_field"
        ),
        openai_realtime_vad_threshold=float(os.getenv("OPENAI_REALTIME_VAD_THRESHOLD", "0.65")),
        openai_realtime_vad_prefix_padding_ms=int(
            os.getenv("OPENAI_REALTIME_VAD_PREFIX_PADDING_MS", "500")
        ),
        openai_realtime_vad_silence_duration_ms=int(
            os.getenv("OPENAI_REALTIME_VAD_SILENCE_DURATION_MS", "700")
        ),
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            (
                "You are a helpful, warm, and concise voice assistant speaking on a phone call. "
                "Answer the caller directly, ask one clear follow-up question when needed, and keep "
                "responses short enough to sound natural when spoken. Always reply in Hinglish using "
                "Roman script only, for example: 'Haan, samajh gaya. Aap thoda aur bata sakte ho?' "
                "Do not use Devanagari or pure English unless the caller explicitly asks for a "
                "different language or script."
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
        bcrm_base_url=os.getenv("BCRM_BASE_URL"),
        internal_api_token=os.getenv("INTERNAL_API_TOKEN"),
        max_concurrent_calls=int(os.getenv("MAX_CONCURRENT_CALLS", "10")),
        call_summary_enabled=_bool_env("CALL_SUMMARY_ENABLED", True),
        call_summary_model=os.getenv("CALL_SUMMARY_MODEL", "gpt-4o-mini"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
