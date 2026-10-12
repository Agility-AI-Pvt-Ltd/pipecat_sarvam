"""Ending a call the moment the conversation is over.

Without this, the AI says its last sentence and the line stays open until the
duration cap — ten minutes of silence for a customer who has just said they were
busy. That is a worse experience than not calling at all: they told us they had no
time and we left them holding a live phone.

So the model is given one way to hang up. It finishes its last message with a
marker, this processor strips the marker before anything reaches TTS, lets the
sentence finish playing, and only then ends the call. The order matters — hanging
up on `LLMFullResponseEndFrame` would cut the thank you off halfway through, which
is rude in a way the customer would remember.

Why a marker and not a tool call: the voice path has no function-calling plumbing,
and on a phone call every round trip is audible. A token at the end of a sentence
is something a small, fast model gets right consistently, and it is cheap to test.

The marker itself, and the stream filter that removes it, live in `end_call.py`
with no Pipecat dependency, so the fiddly half is unit-tested.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndWorkerFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_sarvam_vobiz.end_call import END_CALL_MARKER, MarkerStripper

#: If the closing line somehow never reaches the speaker — a TTS failure, a
#: stream that died — end the call anyway rather than leaving it open. Generous
#: enough that a long goodbye finishes first.
FALLBACK_CLOSE_SECONDS = 20.0


class CallCloser(FrameProcessor):
    """Strips the end-call marker, then ends the call once the bot has finished.

    Sits between the LLM and TTS, which is the one position that can do both jobs:
    the model's text passes through downstream, and `BotStoppedSpeakingFrame` —
    which the output transport pushes in both directions — arrives upstream.
    """

    def __init__(
        self,
        *,
        marker: str = END_CALL_MARKER,
        on_close: Callable[[], Awaitable[None]] | None = None,
        session: object | None = None,
    ) -> None:
        super().__init__(name="call-closer")
        self._marker = marker
        self._on_close = on_close
        self._session = session
        self._stripper = MarkerStripper(marker)
        self._closing = False
        self._closed = False
        self._fallback: asyncio.Task | None = None

    # -- the text -----------------------------------------------------------

    async def _emit(self, text: str, direction: FrameDirection) -> None:
        if text:
            await self.push_frame(LLMTextFrame(text), direction)

    async def _take(self, text: str, direction: FrameDirection) -> None:
        """Pass the model's words through, minus the marker."""
        send = self._stripper.feed(text)
        if self._stripper.found and not self._closing:
            self._closing = True
            logger.info("Call close requested by the assistant")
        await self._emit(send, direction)

    # -- the ending ---------------------------------------------------------

    async def _close(self, reason: str) -> None:
        """Hang up, then end the pipeline. Once per call.

        `EndWorkerFrame` goes downstream, which flushes whatever is still queued
        before the pipeline stops — by this point that is nothing, but it is the
        difference between a clean teardown and cutting the socket.
        """
        if self._closed:
            return
        self._closed = True
        if self._fallback is not None and not self._fallback.done():
            self._fallback.cancel()
        logger.info("Ending the call ({})", reason)
        if self._on_close is not None:
            try:
                await self._on_close()
            except Exception:  # noqa: BLE001 - nothing upstream can act on this
                # The pipeline still ends. A hangup we could not request leaves the
                # call to its duration cap, which is the old behaviour, not a new
                # failure — and there is nobody left to handle an exception here.
                logger.exception("Hangup request failed")
        await self.push_frame(EndWorkerFrame(reason=reason))

    async def _arm_fallback(self) -> None:
        if self._fallback is not None and not self._fallback.done():
            return

        async def _later() -> None:
            try:
                await asyncio.sleep(FALLBACK_CLOSE_SECONDS)
            except asyncio.CancelledError:
                return
            await self._close("closing line never finished playing")

        self._fallback = asyncio.create_task(_later())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._stripper.reset()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            # Swallowed and re-emitted without the marker, so neither the TTS nor
            # the stored transcript ever sees it.
            await self._take(frame.text, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            await self._emit(self._stripper.flush(), direction)
            await self.push_frame(frame, direction)
            if self._closing:
                await self._arm_fallback()
            return

        if isinstance(frame, BotStoppedSpeakingFrame) and self._closing and not self._closed:
            # The goodbye has finished playing. Now the line can go.
            await self.push_frame(frame, direction)
            await self._close("assistant ended the call")
            return

        await self.push_frame(frame, direction)


__all__ = ["FALLBACK_CLOSE_SECONDS", "CallCloser"]
