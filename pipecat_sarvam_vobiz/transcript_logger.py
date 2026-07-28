from __future__ import annotations

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TerminalTranscriptLogger(FrameProcessor):
    """Print streaming STT text to the terminal while keeping frames flowing."""

    def __init__(self) -> None:
        super().__init__(name="terminal-transcript-logger")
        self._last_partial_len = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            text = frame.text.strip()
            if text:
                padded = text + " " * max(0, self._last_partial_len - len(text))
                print(f"\r[partial] {padded}", end="", flush=True)
                self._last_partial_len = len(text)
        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                if self._last_partial_len:
                    print()
                    self._last_partial_len = 0
                print(f"[final] {text}", flush=True)

        await self.push_frame(frame, direction)


class TerminalOpenAILogger(FrameProcessor):
    """Print streaming OpenAI assistant text before it is sent to TTS."""

    def __init__(self) -> None:
        super().__init__(name="terminal-openai-logger")
        self._chunks: list[str] = []
        self._streaming = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._chunks = []
            self._streaming = True
            print("[openai] ", end="", flush=True)
        elif isinstance(frame, LLMTextFrame):
            text = frame.text
            if text:
                if not self._streaming:
                    print("[openai] ", end="", flush=True)
                    self._streaming = True
                self._chunks.append(text)
                print(text, end="", flush=True)
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._streaming:
                full_text = "".join(self._chunks).strip()
                print()
                if full_text:
                    print(f"[openai final] {full_text}", flush=True)
            self._chunks = []
            self._streaming = False

        await self.push_frame(frame, direction)
