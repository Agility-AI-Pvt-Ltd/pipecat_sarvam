"""Turn a finished conversation into something a CRM can act on.

Runs *after* the call, never during it. A summarisation request in the live path
would compete with the Realtime session for the same rate limit and add latency to
the one thing on this box that cannot tolerate any, so this happens once the
socket is closed and the pipeline is torn down.

It is also entirely optional. If no API key is set, or the model returns something
unparseable, the CRM still receives the transcript and the duration — which are
the facts. The outcome is an interpretation, and a missing interpretation is much
better than a confident wrong one written into a dealer's pipeline.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

#: Constrained vocabulary. An open-ended `outcome` string would put free text into
#: a column the CRM filters and reports on, and every call would invent a new
#: value. The model picks one of these or the outcome stays unknown.
OUTCOMES = (
    "interested",
    "not_interested",
    "callback_requested",
    "site_visit_scheduled",
    "wrong_number",
    "voicemail",
    "no_meaningful_conversation",
)

NEXT_ACTIONS = (
    "schedule_site_visit",
    "send_details",
    "call_back_later",
    "assign_to_agent",
    "close_lead",
    "none",
)

_PROMPT = (
    "You are summarising a phone call between an AI real-estate assistant and a "
    "customer. The customer speaks Hinglish. Reply with JSON only, no prose, using "
    "exactly these keys: outcome, next_action, summary.\n"
    f"outcome must be one of: {', '.join(OUTCOMES)}.\n"
    f"next_action must be one of: {', '.join(NEXT_ACTIONS)}.\n"
    "summary must be at most two sentences of plain English describing what the "
    "customer wants and what was agreed. If the call had no real conversation, use "
    "outcome no_meaningful_conversation and next_action none."
)


@dataclass
class CallOutcome:
    outcome: str | None = None
    next_action: str | None = None
    summary: str | None = None


def summarize(
    transcript: str,
    *,
    api_key: str | None,
    model: str,
    base_url: str = "https://api.openai.com/v1",
    timeout: int = 20,
) -> CallOutcome:
    """Classify one transcript. Returns an empty outcome rather than raising."""
    text = (transcript or "").strip()
    if not text or not api_key:
        return CallOutcome()
    # A two-line transcript is hold music and a hangup. Asking a model to find
    # intent in it produces invention, so it is classified here instead.
    if len(text) < 40:
        return CallOutcome(
            outcome="no_meaningful_conversation",
            next_action="none",
            summary="Call connected but no meaningful conversation took place.",
        )

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _PROMPT},
            {"role": "user", "content": text[:12000]},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
    ) as exc:
        print(f"[outcome] summarisation failed: {exc}", flush=True)
        return CallOutcome()

    outcome = str(parsed.get("outcome") or "").strip()
    next_action = str(parsed.get("next_action") or "").strip()
    return CallOutcome(
        outcome=outcome if outcome in OUTCOMES else None,
        next_action=next_action if next_action in NEXT_ACTIONS else None,
        summary=(str(parsed.get("summary") or "").strip() or None),
    )
