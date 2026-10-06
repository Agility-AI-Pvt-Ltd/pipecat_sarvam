"""Speak the caller's language.

Sarvam STT runs with language auto-detection, so every final transcript arrives
tagged with the language the caller used. This processor watches those tags and,
when the caller has settled on a language, tells the TTS to switch voice language
to match. The LLM needs no help: it reads the caller's words (Tamil arrives in
Tamil script) and its prompt tells it to answer in kind.

Hindi and English are one voice family on purpose. The prompt has the bot write
Hindi and Hinglish in Roman script, which an `en-IN` voice would read as English
and mangle; `hi-IN` reads both Roman Hinglish and plain English acceptably. So
English never pulls the voice away from Hindi, and a caller drifting between the
two — most callers — does not make the voice flap.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame, TTSUpdateSettingsFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transcriptions.language import Language

#: Sarvam language code -> the code the TTS should use for it.
VOICE_FOR: dict[str, str] = {
    "hi-IN": "hi-IN",
    "en-IN": "hi-IN",
    "bn-IN": "bn-IN",
    "gu-IN": "gu-IN",
    "kn-IN": "kn-IN",
    "ml-IN": "ml-IN",
    "mr-IN": "mr-IN",
    "od-IN": "od-IN",
    "or-IN": "od-IN",
    "pa-IN": "pa-IN",
    "ta-IN": "ta-IN",
    "te-IN": "te-IN",
}

_ENUM_FOR: dict[str, Language] = {
    "hi-IN": Language.HI_IN,
    "bn-IN": Language.BN_IN,
    "gu-IN": Language.GU_IN,
    "kn-IN": Language.KN_IN,
    "ml-IN": Language.ML_IN,
    "mr-IN": Language.MR_IN,
    "od-IN": Language.OR_IN,
    "pa-IN": Language.PA_IN,
    "ta-IN": Language.TA_IN,
    "te-IN": Language.TE_IN,
}

#: Consecutive transcripts in a new language before the voice follows. One stray
#: misdetection — a name, a place, a single English word — must not switch it.
CONFIRMATIONS = 2


def voice_code(language: Language | str | None) -> str | None:
    """The TTS language for a detected STT language, or None if unsupported."""
    if language is None:
        return None
    raw = getattr(language, "value", language)
    text = str(raw).strip()
    if not text:
        return None
    if "-" not in text:
        text = f"{text.lower()}-IN"
    else:
        head, _, tail = text.partition("-")
        text = f"{head.lower()}-{tail.upper()}"
    return VOICE_FOR.get(text)


@dataclass
class LanguageState:
    current: str
    candidate: str | None = None
    streak: int = 0
    heard_any: bool = False


def observe(state: LanguageState, detected: str | None) -> str | None:
    """Feed one final transcript's language. Returns a new voice code to switch
    to, or None to keep the current one. Mutates `state`.

    The very first transcript switches at once: the caller's opening "Hello?" /
    "Vanakkam" is the best signal of the call, and waiting a turn would make the
    bot's first reply come out in the wrong language.
    """
    if detected is None:
        return None
    first = not state.heard_any
    state.heard_any = True
    if detected == state.current:
        state.candidate, state.streak = None, 0
        return None
    if first:
        state.current, state.candidate, state.streak = detected, None, 0
        return detected
    if detected == state.candidate:
        state.streak += 1
    else:
        state.candidate, state.streak = detected, 1
    if state.streak >= CONFIRMATIONS:
        state.current, state.candidate, state.streak = detected, None, 0
        return detected
    return None


class CallerLanguageFollower(FrameProcessor):
    """Sits right after the STT; switches the TTS voice language to the caller's."""

    def __init__(self, *, initial: str, tts: SarvamTTSService, session=None, **kwargs):
        super().__init__(**kwargs)
        self._state = LanguageState(current=voice_code(initial) or "hi-IN")
        self._tts = tts
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            detected = voice_code(frame.language)
            if self._session is not None and detected:
                self._session.language = detected
            switch_to = observe(self._state, detected)
            if switch_to:
                logger.info(f"Caller language -> {switch_to}; switching TTS voice language")
                # Pushed before the transcript, so it reaches the TTS ahead of the
                # reply to this very utterance.
                await self.push_frame(
                    TTSUpdateSettingsFrame(
                        delta=SarvamTTSService.Settings(language=_ENUM_FOR[switch_to]),
                        service=self._tts,
                    ),
                    direction,
                )
        await self.push_frame(frame, direction)


__all__ = ["CallerLanguageFollower", "LanguageState", "observe", "voice_code"]
