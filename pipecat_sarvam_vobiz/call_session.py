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
and 1,000 Sarvam sockets on one box, and every one of them degrades. It is
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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

Closer = Callable[[], Awaitable[None]]


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
    #: The voice language the caller settled on (e.g. "ta-IN"), as detected by
    #: Sarvam STT. Reported with the outcome.
    language: str | None = None
    started_at: float = field(default_factory=time.time)
    answered_at: float | None = None
    last_activity_at: float = field(default_factory=time.time)
    stream_started: bool = False
    #: False for browser-test / smoke / random internet sockets. They may still
    #: run a pipeline, but they must not fill the admission ceiling.
    counts_toward_capacity: bool = True
    disconnect: Closer | None = None
    on_provider_id: Callable[[str], None] | None = None
    turns: list[TranscriptTurn] = field(default_factory=list)

    def touch(self) -> None:
        self.last_activity_at = time.time()

    def note_stream(self) -> None:
        self.stream_started = True
        self.answered_at = self.answered_at or time.time()
        self.touch()

    def record(self, role: str, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        # A reply can arrive as several LLM responses (an interruption, a retry)
        # and a customer's sentence as several final transcripts when they pause
        # mid-thought. Joining consecutive turns from the same speaker keeps the
        # transcript readable instead of one fragment a line.
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

    Live sockets also expire. Vobiz (and the public tunnel in front of it) often
    leaves the media WebSocket open after the PSTN hangup. Without an idle reap,
    those sockets hold the ceiling forever and the next real pickup gets Hangup
    XML. Connect-idle covers sockets that never received a `start` event;
    media-idle covers a stream that has gone silent.
    """

    def __init__(
        self,
        *,
        max_concurrent: int,
        reservation_ttl_seconds: int = 60,
        connect_idle_seconds: int = 30,
        media_idle_seconds: int = 45,
    ) -> None:
        self._max = max(0, int(max_concurrent))
        self._ttl = max(5, int(reservation_ttl_seconds))
        self._connect_idle = max(5, int(connect_idle_seconds))
        self._media_idle = max(10, int(media_idle_seconds))
        self._lock = threading.Lock()
        self._reservations: dict[str, float] = {}
        self._active: dict[str, CallSession] = {}
        self._probes: dict[str, CallSession] = {}
        self._by_vobiz_id: dict[str, str] = {}

    @property
    def limit(self) -> int:
        return self._max

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            self._expire_locked()
            active = len(self._active)
            reserved = len(self._reservations)
            probes = len(self._probes)
        return {
            "active_calls": active,
            "reserved_slots": reserved,
            "probe_sockets": probes,
            "capacity": self._max,
            "accepting": self._max <= 0 or (active + reserved) < self._max,
        }

    def reserve(self, call_id: str | None) -> str:
        """Claim a slot for a call that is about to connect.

        Raises `CapacityError` when full, so the answer handler can return XML
        that hangs up politely instead of accepting a call it cannot serve.

        A request with no `call_id` is a probe or the browser test client. It
        still gets a token so the WebSocket can be tracked, but it does not take
        a capacity slot.
        """
        if not call_id:
            return f"probe-{uuid.uuid4().hex[:12]}"
        token = call_id
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
        session.touch()
        session.counts_toward_capacity = bool(session.call_id) and session.counts_toward_capacity
        with self._lock:
            self._reservations.pop(token, None)
            if session.counts_toward_capacity:
                self._active[token] = session
                self._probes.pop(token, None)
            else:
                self._probes[token] = session
                self._active.pop(token, None)

    def bind_vobiz_id(self, token: str, vobiz_call_id: str | None) -> None:
        provider_id = (vobiz_call_id or "").strip()
        if not provider_id:
            return
        with self._lock:
            self._by_vobiz_id[provider_id] = token

    def release(self, token: str) -> None:
        with self._lock:
            self._reservations.pop(token, None)
            self._active.pop(token, None)
            self._probes.pop(token, None)
            stale = [key for key, value in self._by_vobiz_id.items() if value == token]
            for key in stale:
                self._by_vobiz_id.pop(key, None)

    def closer_for(self, *idents: str | None) -> Closer | None:
        """The disconnect callback for a CRM call id or a Vobiz CallUUID."""
        with self._lock:
            session = self._session_locked(*idents)
            return session.disconnect if session is not None else None

    def pop_idle(self) -> list[tuple[str, Closer | None]]:
        """Sockets that have gone silent and should be closed."""
        now = time.time()
        idle: list[tuple[str, Closer | None]] = []
        with self._lock:
            self._expire_locked()
            for token, session in list(self._active.items()) + list(self._probes.items()):
                limit = self._media_idle if session.stream_started else self._connect_idle
                if now - session.last_activity_at >= limit:
                    idle.append((token, session.disconnect))
        return idle

    def _session_locked(self, *idents: str | None) -> CallSession | None:
        for ident in idents:
            token = (ident or "").strip()
            if not token:
                continue
            if token in self._active:
                return self._active[token]
            if token in self._probes:
                return self._probes[token]
            mapped = self._by_vobiz_id.get(token)
            if mapped and mapped in self._active:
                return self._active[mapped]
            if mapped and mapped in self._probes:
                return self._probes[mapped]
        return None

    def _expire_locked(self) -> None:
        now = time.time()
        stale = [key for key, expiry in self._reservations.items() if expiry < now]
        for key in stale:
            self._reservations.pop(key, None)
