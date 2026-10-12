"""Stripping the end-call marker out of a token stream.

The processor itself needs Pipecat; this is the part that can go wrong silently,
so it is pure and tested here. Two failures matter:

* leaving the marker in the text, so the customer hears "double bracket end call"
  read aloud as the last thing on the call;
* missing it because the model's tokens split it in half, so the AI says thank you
  and then leaves the line open for ten minutes.
"""

from __future__ import annotations

import unittest

MARKER = "[[END_CALL]]"


def _stripper():
    from pipecat_sarvam_vobiz.end_call import MarkerStripper

    return MarkerStripper(MARKER)


def _run(chunks: list[str]) -> tuple[str, bool]:
    """Feed a stream through and return `(spoken text, close requested)`."""
    stripper = _stripper()
    out = "".join(stripper.feed(chunk) for chunk in chunks)
    return out + stripper.flush(), stripper.found


class MarkerStripperTests(unittest.TestCase):
    def test_a_stream_without_the_marker_passes_through_unchanged(self) -> None:
        text, found = _run(["Namaste", " ji", "! Kya", " haal hai?"])
        self.assertEqual(text, "Namaste ji! Kya haal hai?")
        self.assertFalse(found)

    def test_the_marker_is_removed_and_the_close_is_requested(self) -> None:
        text, found = _run(["Ji bilkul, thank you! ", MARKER])
        self.assertEqual(text, "Ji bilkul, thank you! ")
        self.assertTrue(found)

    def test_a_marker_split_across_chunks_is_still_found(self) -> None:
        """The one that bites. A model streams tokens, not sentences, so the
        marker arrives in pieces — and a stripper that only matched whole chunks
        would read it out loud and never hang up."""
        for split in range(1, len(MARKER)):
            chunks = ["Thank you! ", MARKER[:split], MARKER[split:]]
            text, found = _run(chunks)
            self.assertEqual(text, "Thank you! ", f"split at {split}")
            self.assertTrue(found, f"split at {split}")

    def test_a_marker_arriving_one_character_at_a_time_is_still_found(self) -> None:
        text, found = _run(["Bye! ", *list(MARKER)])
        self.assertEqual(text, "Bye! ")
        self.assertTrue(found)

    def test_a_marker_in_the_middle_keeps_the_text_either_side(self) -> None:
        text, found = _run([f"Thank you{MARKER} ji"])
        self.assertEqual(text, "Thank you ji")
        self.assertTrue(found)

    def test_the_marker_twice_is_stripped_twice(self) -> None:
        text, found = _run([f"Bye{MARKER}{MARKER}"])
        self.assertEqual(text, "Bye")
        self.assertTrue(found)

    def test_text_is_not_held_back_waiting_for_a_marker_that_never_comes(self) -> None:
        """Latency matters on a phone call: only the last few characters wait."""
        stripper = _stripper()
        spoken = stripper.feed("Aapka budget kitna hai, aur kaunsa area pasand hai?")
        self.assertTrue(spoken.startswith("Aapka budget kitna hai"))
        self.assertLessEqual(
            len("Aapka budget kitna hai, aur kaunsa area pasand hai?") - len(spoken),
            len(MARKER) - 1,
        )

    def test_nothing_is_lost_however_the_stream_is_chopped(self) -> None:
        sentence = "Theek hai ji, main aapko WhatsApp par details bhej deta hoon. Dhanyavaad!"
        for size in (1, 2, 3, 5, 11, 12, 13, 40):
            chunks = [sentence[i : i + size] for i in range(0, len(sentence), size)]
            text, found = _run(chunks)
            self.assertEqual(text, sentence, f"chunk size {size}")
            self.assertFalse(found, f"chunk size {size}")

    def test_a_new_response_does_not_withdraw_a_requested_close(self) -> None:
        """Once the model has said the call is over, a stray extra response must
        not cancel the hangup — otherwise an interruption leaves the line open."""
        from pipecat_sarvam_vobiz.end_call import MarkerStripper

        stripper = MarkerStripper(MARKER)
        stripper.feed(f"Thank you! {MARKER}")
        stripper.flush()
        stripper.reset()
        stripper.feed("Haan ji?")
        self.assertTrue(stripper.found)

    def test_an_empty_marker_is_refused(self) -> None:
        from pipecat_sarvam_vobiz.end_call import MarkerStripper

        with self.assertRaises(ValueError):
            MarkerStripper("")

    def test_the_marker_matches_the_one_the_crm_puts_in_the_prompt(self) -> None:
        """The two live in separate repositories. A drift means no hangup, which is
        survivable — but it should be caught here rather than on a call."""
        from pipecat_sarvam_vobiz.end_call import END_CALL_MARKER

        self.assertEqual(END_CALL_MARKER, "[[END_CALL]]")


if __name__ == "__main__":
    unittest.main()
