"""The end-call marker, and taking it out of a token stream.

Kept free of Pipecat and of loguru on purpose: this is the part that can fail
silently, so it is a unit test rather than something discovered on a live call.
`call_closer.py` is the thin adapter that wires it into the pipeline.
"""

from __future__ import annotations

#: How the model says "this call is finished, hang up".
#:
#: A marker in the text rather than a tool call, because the voice path has no
#: function-calling plumbing and on a phone call every round trip is audible. A
#: token at the end of a sentence is something a small, fast model gets right
#: consistently.
#:
#: Must equal `END_CALL_MARKER` in the CRM's `app/modules/calls/enquiry_policy.py`,
#: which is what puts it into the prompt. If the two ever drift the marker is
#: simply never recognised: the call runs to its own timeout as it used to, rather
#: than ending mid-sentence.
END_CALL_MARKER = "[[END_CALL]]"


class MarkerStripper:
    """Removes a marker from a token stream without delaying it.

    Pure, and separate from the processor, so the fiddly part is a unit test
    rather than something you discover on a live call. The marker arrives inside a
    stream of LLM tokens and can be split anywhere — `[[END`, `_CALL]]` — so a
    tail of `len(marker) - 1` characters is held back on every chunk. That is
    enough for any split to be reassembled, and eleven characters of latency is
    inaudible on a phone call.

    `found` latches: once the marker has been seen, the stream is a closing one
    whatever arrives afterwards.
    """

    def __init__(self, marker: str) -> None:
        if not marker:
            raise ValueError("a marker is required")
        self._marker = marker
        self._keep = len(marker) - 1
        self._tail = ""
        self.found = False

    def reset(self) -> None:
        """Start a new response. `found` is deliberately not cleared: a close
        asked for in one response is not withdrawn by the next one starting."""
        self._tail = ""

    def feed(self, text: str) -> str:
        """Text safe to pass on now."""
        self._tail += text or ""
        while self._marker in self._tail:
            self.found = True
            self._tail = self._tail.replace(self._marker, "", 1)
        if len(self._tail) <= self._keep:
            return ""
        send, self._tail = self._tail[: -self._keep or None], self._tail[-self._keep :]
        return send

    def flush(self) -> str:
        """Whatever is still held back, at the end of a response."""
        held, self._tail = self._tail, ""
        return held


__all__ = ["END_CALL_MARKER", "MarkerStripper"]
