"""Per-call state and the admission controller.

Two things live here, and the distinction between them is the whole design:

`CallSession` is *per call*. Every WebSocket gets its own instance, holding its own
transcript and its own timings. Nothing about a call is stored in a module-level
variable, which is what makes running three replicas behind a load balancer
possible: a call's state travels with its socket, so no replica needs to know
anything about a call it is not carrying.

`CallRegistry` is *per process*. It is the one piece of shared state, and it exists
to answer a single question: may this process accept one more call? Without it, a
campaign that dials 500 leads gets 500 accepted WebSockets, 500 Pipecat pipelines
and 500 OpenAI Realtime sessions on one box, and every one of them degrades. It is
better to refuse the 21st call cleanly than to ruin twenty conversations.

The capacity number itself is not knowledge this code has. `MAX_CONCURRENT_CALLS`
must come from load testing on the hardware you actually run — the binding
constraint is whichever of Pipecat's CPU cost, the AI provider's rate limits and
the Vobiz account's concurrency cap is lowest, and only measurement finds it.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class TranscriptTurn:
    role: str
    text: str
    at: float = field(default_factory=time.time)


@dataclass
class CallSession:
    """Everything one live call knows about itself."""

    call_id: str | None = None
    stream_id: str | None = None
    vobiz_call_id: str | None = None
    #: Filled from the CRM's call context on connect, so the AI can greet the
    #: person by name and the transcript is attributable when it is reviewed.
    customer_name: str | None = None
    started_at: float = field(default_factory=time.time)
    answered_at: float | None = None
    turns: list[TranscriptTurn] = field(default_factory=list)

    def record(self, role: str, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        # Realtime emits assistant text in fragments; joining them into the
        # previous turn keeps the transcript readable instead of one word a line.
        if self.turns and self.turns[-1].role == role:
            self.turns[-1].text = f"{self.turns[-1].text} {cleaned}".strip()
            return
        self.turns.append(TranscriptTurn(role=role, text=cleaned))

    def transcript_text(self) -> str:
        labels = {"user": "Customer", "assistant": "Agent"}
        return "\n".join(f"{labels.get(t.role, t.role)}: {t.text}" for t in self.turns)

    def duration_seconds(self) -> int:
        start = self.answered_at or self.started_at
        return max(0, int(time.time() - start))


class CapacityError(RuntimeError):
    """Raised when this process cannot take another call."""


class CallRegistry:
    """Counts live and reserved calls, and refuses the ones over the line.

    Slots are reserved at the answer URL rather than at WebSocket connect,
    because Vobiz fetches the XML first and only then opens the stream. Counting
    sockets alone would let a burst of fifty answers all pass the check before any
    of them became a socket, and the ceiling would mean nothing under exactly the
    load it exists to handle.

    Reservations therefore expire: if Vobiz fetches the XML and the call dies
    before the stream opens — the customer hangs up mid-ring — nothing would ever
    release that slot, and the process would slowly refuse to work.
    """

    def __init__(self, *, max_concurrent: int, reservation_ttl_seconds: int = 60) -> None:
        self._max = max(0, int(max_concurrent))
        self._ttl = max(5, int(reservation_ttl_seconds))
        self._lock = threading.Lock()
        self._reservations: dict[str, float] = {}
        self._active: dict[str, CallSession] = {}

    @property
    def limit(self) -> int:
        return self._max

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            self._expire_locked()
            active = len(self._active)
            reserved = len(self._reservations)
        return {
            "active_calls": active,
            "reserved_slots": reserved,
            "capacity": self._max,
            "accepting": self._max <= 0 or (active + reserved) < self._max,
        }

    def reserve(self, call_id: str | None) -> str:
        """Claim a slot for a call that is about to connect.

        Raises `CapacityError` when full, so the answer handler can return XML
        that hangs up politely instead of accepting a call it cannot serve.
        """
        token = call_id or f"anon-{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._expire_locked()
            if self._max > 0 and (len(self._active) + len(self._reservations)) >= self._max:
                raise CapacityError(
                    f"at capacity: {len(self._active)} active, "
                    f"{len(self._reservations)} reserved, limit {self._max}"
                )
            self._reservations[token] = time.time() + self._ttl
        return token

    def activate(self, token: str, session: CallSession) -> None:
        """The stream opened: turn the reservation into a live call."""
        with self._lock:
            self._reservations.pop(token, None)
            self._active[token] = session

    def release(self, token: str) -> None:
        with self._lock:
            self._reservations.pop(token, None)
            self._active.pop(token, None)

    def _expire_locked(self) -> None:
        now = time.time()
        stale = [key for key, expiry in self._reservations.items() if expiry < now]
        for key in stale:
            self._reservations.pop(key, None)
