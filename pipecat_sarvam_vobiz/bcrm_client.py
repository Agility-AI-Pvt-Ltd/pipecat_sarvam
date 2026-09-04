"""Report call results back to the CRM.

This is the only thing this service knows about the outside world, and the
vocabulary is deliberately tiny:

    "call X was answered"
    "call X ended, and here is what happened"

No tenants, no campaigns, no leads, no nurture rules, no Temporal. The CRM owns
all of that and decides what an outcome means. Keeping the conversation this
narrow is what lets the same Voice Agent serve a different product without
changes — and it means an orchestration change on the CRM side never reaches the
audio path.

Every call here is best-effort. A failed report must never take down a live call
or delay a hangup: the CRM also receives Vobiz's own `Hangup` callback, so the
call is closed out correctly even when this never arrives. What is lost is the
transcript and the outcome, which is worth a loud log line and nothing more.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class BcrmClient:
    def __init__(
        self,
        *,
        base_url: str | None,
        token: str | None,
        timeout: int = 10,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.token = token or ""
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def fetch_context(self, call_id: str) -> dict[str, Any] | None:
        """Ask the CRM how this particular call should be handled.

        This is the one place the vocabulary widens, and it earns it: without it
        the conversation prompt has to live in this service's environment, which
        means every dealer on the deployment shares one script and changing it is
        a redeploy. Fetching per call is what lets a dealer write their own script
        in the CRM and have the very next call use it.

        Returns None on any failure, which the caller treats as "use the default
        prompt". A CRM outage should make the AI generic, never mute.
        """
        return self._get(f"/api/v1/internal/calls/{call_id}/context")

    def report_answered(self, call_id: str) -> bool:
        """Tell the CRM the customer picked up.

        Vobiz delivers `StartApp` to the answer URL — this service — so the CRM
        has no other way to learn that a call was actually answered rather than
        merely ringing.
        """
        return self._post(f"/api/v1/internal/calls/{call_id}/answered", {}) is not None

    def report_outcome(
        self,
        call_id: str,
        *,
        outcome: str | None = None,
        next_action: str | None = None,
        summary: str | None = None,
        transcript: str | None = None,
        duration_seconds: int | None = None,
        recording_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        body: dict[str, Any] = {"metadata": metadata or {}}
        if outcome:
            body["outcome"] = outcome
        if next_action:
            body["next_action"] = next_action
        if summary:
            body["summary"] = summary
        if transcript:
            body["transcript"] = transcript
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds
        if recording_url:
            body["recording_url"] = recording_url
        return self._post(f"/api/v1/internal/calls/{call_id}/outcome", body) is not None

    def _get(self, path: str) -> dict[str, Any] | None:
        if not self.configured:
            print(
                f"[bcrm] skipped {path}: BCRM_BASE_URL / INTERNAL_API_TOKEN not set",
                flush=True,
            )
            return None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"X-Internal-Token": self.token},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8") or "{}")
            print(f"[bcrm] {path} ok", flush=True)
            # The CRM wraps every response in {success, data, error}.
            data = payload.get("data") if isinstance(payload, dict) else None
            return data if isinstance(data, dict) else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            print(f"[bcrm] {path} failed status={exc.code} detail={detail}", flush=True)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"[bcrm] {path} failed reason={exc}", flush=True)
        return None

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any] | None:
        if not self.configured:
            print(
                f"[bcrm] skipped {path}: BCRM_BASE_URL / INTERNAL_API_TOKEN not set",
                flush=True,
            )
            return None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Internal-Token": self.token,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8") or "{}"
            print(f"[bcrm] {path} ok", flush=True)
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            print(f"[bcrm] {path} failed status={exc.code} detail={detail}", flush=True)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"[bcrm] {path} failed reason={exc}", flush=True)
        return None
