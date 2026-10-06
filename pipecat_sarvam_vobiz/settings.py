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

    # --- the LLM that does the thinking ------------------------------------
    # Any OpenAI-compatible chat-completions endpoint. Defaults to OpenRouter,
    # which is what the `OPENAI_*` names now point at: the variables keep their
    # names because every OpenAI-compatible gateway reads them, not because
    # OpenAI is involved. Speech is Sarvam on both sides; this is text in, text out.
    openai_api_key: str | None
    openai_model: str
    openai_base_url: str
    openai_temperature: float
    openai_max_tokens: int
    system_prompt: str
    public_base_url: str | None
    vobiz_ws_url: str | None
    vobiz_stream_content_type: str
    sarvam_mode: str
    sarvam_language: str | None
    sarvam_tts_model: str
    sarvam_tts_voice: str
    sarvam_tts_language: str
    sarvam_tts_pace: float
    sarvam_tts_min_buffer_size: int
    sarvam_tts_max_chunk_length: int

    # --- turn-taking: Silero VAD, run locally ------------------------------
    # Sarvam's streaming STT can report speech start/stop itself
    # (`vad_signals`), but every boundary then costs a network round trip, on
    # the one path where latency is audible: deciding the caller has finished,
    # and noticing they have started talking over the bot. Silero runs on this
    # box in a few milliseconds per frame, and when it says the caller stopped,
    # Pipecat tells Sarvam to flush the transcript. Only one of them may decide —
    # two VADs means two interruption signals for one breath.
    vad_confidence: float
    vad_start_secs: float
    vad_stop_secs: float
    vad_min_volume: float
    #: How long the caller may pause, after Silero reports silence, before the
    #: turn is treated as finished. Total silence before the bot answers is
    #: roughly `vad_stop_secs + user_turn_silence_secs`.
    user_turn_silence_secs: float
    #: Whether the caller can interrupt the bot. The bot's very first reply is
    #: always protected — see `agent.build_user_mute_strategies`.
    barge_in: bool
    #: Pause between pickup and the opening greeting.
    greeting_delay_secs: float
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
    #: Model for the post-call summary. Defaults to `openai_model`: a model id is
    #: only meaningful to the gateway it was written for, and the old default
    #: `gpt-4o-mini` does not exist on OpenRouter (there it is `openai/gpt-4o-mini`),
    #: so summaries failed quietly whenever the gateway changed.
    call_summary_model: str

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
    openai_model = os.getenv("OPENAI_MODEL", "openrouter/free")
    return Settings(
        sarvam_api_key=os.getenv("SARVAM_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=openai_model,
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.4")),
        # Kept under its old name so existing .env files carry over. Phone replies
        # must be short; a long answer is dead air while TTS catches up.
        openai_max_tokens=int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", "180")),
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            (
                "You are a helpful, warm, and concise voice assistant speaking on a phone call. "
                "Answer the caller directly, ask one clear follow-up question when needed, and keep "
                "responses short enough to sound natural when spoken. Always reply in Hinglish using "
                "Roman script only, for example: 'Haan, samajh gaya. Aap thoda aur bata sakte ho?' "
                "Do not use Devanagari or pure English unless the caller explicitly asks for a "
                "different language or script. Never use markdown, lists, emoji or symbols: "
                "everything you write is read aloud."
            ),
        ),
        public_base_url=os.getenv("PUBLIC_BASE_URL"),
        vobiz_ws_url=os.getenv("VOBIZ_WS_URL"),
        vobiz_stream_content_type=os.getenv(
            "VOBIZ_STREAM_CONTENT_TYPE", "audio/x-l16;rate=16000"
        ),
        sarvam_mode=os.getenv("SARVAM_MODE", "transcribe"),
        sarvam_language=os.getenv("SARVAM_LANGUAGE"),
        sarvam_tts_model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"),
        sarvam_tts_voice=os.getenv("SARVAM_TTS_VOICE", "shubh"),
        sarvam_tts_language=os.getenv("SARVAM_TTS_LANGUAGE", "hi-IN"),
        sarvam_tts_pace=float(os.getenv("SARVAM_TTS_PACE", "1.05")),
        sarvam_tts_min_buffer_size=int(os.getenv("SARVAM_TTS_MIN_BUFFER_SIZE", "20")),
        sarvam_tts_max_chunk_length=int(os.getenv("SARVAM_TTS_MAX_CHUNK_LENGTH", "120")),
        vad_confidence=float(os.getenv("VAD_CONFIDENCE", "0.7")),
        vad_start_secs=float(os.getenv("VAD_START_SECS", "0.2")),
        vad_stop_secs=float(os.getenv("VAD_STOP_SECS", "0.2")),
        vad_min_volume=float(os.getenv("VAD_MIN_VOLUME", "0.6")),
        user_turn_silence_secs=float(os.getenv("USER_TURN_SILENCE_SECS", "0.6")),
        barge_in=_bool_env("BARGE_IN", True),
        greeting_delay_secs=float(os.getenv("GREETING_DELAY_SECS", "0.6")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        bcrm_base_url=os.getenv("BCRM_BASE_URL"),
        internal_api_token=os.getenv("INTERNAL_API_TOKEN"),
        max_concurrent_calls=int(os.getenv("MAX_CONCURRENT_CALLS", "10")),
        call_summary_enabled=_bool_env("CALL_SUMMARY_ENABLED", True),
        call_summary_model=os.getenv("CALL_SUMMARY_MODEL") or openai_model,
    )
