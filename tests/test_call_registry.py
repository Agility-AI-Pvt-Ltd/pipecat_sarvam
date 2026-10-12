from __future__ import annotations

import time
import unittest

from pipecat_sarvam_vobiz.call_session import CallRegistry, CallSession, CapacityError


class CallRegistryTests(unittest.TestCase):
    def test_probe_answer_does_not_consume_capacity(self) -> None:
        registry = CallRegistry(max_concurrent=1)
        probe = registry.reserve(None)
        real = registry.reserve("call-1")
        self.assertTrue(probe.startswith("probe-"))
        self.assertEqual(real, "call-1")
        self.assertEqual(registry.snapshot()["active_calls"], 0)
        self.assertEqual(registry.snapshot()["reserved_slots"], 1)
        self.assertFalse(registry.snapshot()["accepting"])

    def test_probe_socket_does_not_block_a_real_call(self) -> None:
        registry = CallRegistry(max_concurrent=1)
        probe = CallSession(call_id=None, counts_toward_capacity=False)
        registry.activate("probe-1", probe)
        self.assertEqual(registry.snapshot()["active_calls"], 0)
        self.assertEqual(registry.snapshot()["probe_sockets"], 1)
        self.assertTrue(registry.snapshot()["accepting"])
        registry.reserve("call-1")
        self.assertEqual(registry.snapshot()["reserved_slots"], 1)

    def test_full_registry_refuses_another_real_call(self) -> None:
        registry = CallRegistry(max_concurrent=1)
        registry.reserve("call-1")
        with self.assertRaises(CapacityError):
            registry.reserve("call-2")

    def test_idle_connect_is_reaped(self) -> None:
        registry = CallRegistry(max_concurrent=2, connect_idle_seconds=5, media_idle_seconds=45)
        session = CallSession(call_id="call-1")
        registry.activate("call-1", session)
        session.last_activity_at = time.time() - 30
        idle = registry.pop_idle()
        self.assertEqual([token for token, _ in idle], ["call-1"])

    def test_live_media_is_not_reaped(self) -> None:
        registry = CallRegistry(max_concurrent=2, connect_idle_seconds=5, media_idle_seconds=45)
        session = CallSession(call_id="call-1")
        registry.activate("call-1", session)
        session.note_stream()
        self.assertEqual(registry.pop_idle(), [])

    def test_silent_stream_is_reaped(self) -> None:
        registry = CallRegistry(max_concurrent=2, connect_idle_seconds=5, media_idle_seconds=10)
        session = CallSession(call_id="call-1")
        registry.activate("call-1", session)
        session.note_stream()
        session.last_activity_at = time.time() - 30
        idle = registry.pop_idle()
        self.assertEqual([token for token, _ in idle], ["call-1"])

    def test_hangup_finds_session_by_vobiz_id(self) -> None:
        registry = CallRegistry(max_concurrent=2)

        async def disconnect() -> None:
            return None

        session = CallSession(call_id="crm-1", disconnect=disconnect)
        registry.activate("crm-1", session)
        registry.bind_vobiz_id("crm-1", "vobiz-uuid")
        self.assertIs(registry.closer_for("vobiz-uuid"), disconnect)
        self.assertIs(registry.closer_for("crm-1"), disconnect)

    def test_release_frees_the_slot(self) -> None:
        registry = CallRegistry(max_concurrent=1)
        registry.reserve("call-1")
        registry.activate("call-1", CallSession(call_id="call-1"))
        self.assertFalse(registry.snapshot()["accepting"])
        registry.release("call-1")
        self.assertTrue(registry.snapshot()["accepting"])
        self.assertEqual(registry.snapshot()["active_calls"], 0)


if __name__ == "__main__":
    unittest.main()
