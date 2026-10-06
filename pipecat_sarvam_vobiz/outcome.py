"""Turn a finished conversation into something a CRM can act on.

Runs *after* the call, never during it. A summarisation request in the live path
would share a rate limit with the conversation itself and add latency to the one
thing on this box that cannot tolerate any, so this happens once the socket is
closed and the pipeline is torn down.

It is also entirely optional. If no API key is set, or the model returns something
unparseable, the CRM still receives the transcript and the duration — which are
the facts. The outcome is an interpretation, and a missing interpretation is much
better than a confident wrong one written into a dealer's pipeline.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

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

#: What a call can collect about the property the customer wants. Free text in
#: English, as the customer put it — the CRM parses budgets itself.
REQUIREMENT_KEYS = ("budget", "bhk", "locality", "move_in", "purpose")

_PROMPT = (
    "You are summarising a phone call between an AI real-estate assistant and a "
    "customer. The customer may speak Hindi, English, Hinglish or another Indian "
    "language. Reply with JSON only, no prose, using exactly these keys: outcome, "
    "next_action, summary, requirements.\n"
    f"outcome must be one of: {', '.join(OUTCOMES)}.\n"
    f"next_action must be one of: {', '.join(NEXT_ACTIONS)}.\n"
    "summary must be at most two sentences of plain English describing what the "
    "customer wants and what was agreed.\n"
    "requirements must be an object with keys budget, bhk, locality, move_in, purpose. "
    "Each value is a short English phrase in the customer's own terms (for example "
    "\"80 lakh to 1 crore\", \"3 BHK\", \"Sector 62, Noida\", \"within 3 months\", "
    "\"buy\" or \"rent\"), or null if the customer did not say it. Never guess.\n"
    "If the call had no real conversation, use outcome no_meaningful_conversation, "
    "next_action none and all requirements null."
)


@dataclass
class CallOutcome:
    outcome: str | None = None
    next_action: str | None = None
    summary: str | None = None
    #: Only the keys the customer actually gave; empty when none.
    requirements: dict[str, str] = field(default_factory=dict)


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
        parsed = _extract_json_object(content)
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

    if not isinstance(parsed, dict):
        print("[outcome] summarisation returned no JSON object", flush=True)
        return CallOutcome()

    outcome = str(parsed.get("outcome") or "").strip()
    next_action = str(parsed.get("next_action") or "").strip()
    return CallOutcome(
        outcome=outcome if outcome in OUTCOMES else None,
        next_action=next_action if next_action in NEXT_ACTIONS else None,
        summary=(str(parsed.get("summary") or "").strip() or None),
        requirements=_requirements(parsed.get("requirements")),
    )


_EMPTY = {"", "null", "none", "unknown", "n/a", "na", "-"}


def _requirements(value: object) -> dict[str, str]:
    """The known keys with real values, each trimmed to a sane length."""
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, str] = {}
    for key in REQUIREMENT_KEYS:
        raw = value.get(key)
        if raw is None or isinstance(raw, (dict, list)):
            continue
        text = " ".join(str(raw).split())[:80]
        if text.lower() not in _EMPTY:
            cleaned[key] = text
    return cleaned


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json_object(content: str | None) -> dict | None:
    """The JSON object in a reply, tolerating the ways models decorate it.

    `response_format: json_object` is a request, not a guarantee. Behind
    `openrouter/free` each call may land on a different model, and several wrap
    the answer in a ```json fence or put a sentence before it. `json.loads` on that
    raises, and the summary used to be thrown away for formatting alone — the CRM
    lost an outcome the model had in fact produced.

    Strict where it matters: the result must still be one JSON object, and the
    values are checked against `OUTCOMES` / `NEXT_ACTIONS` by the caller, so
    loosening the wrapper does not loosen what is accepted.
    """
    text = _FENCE.sub("", (content or "").strip())
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
