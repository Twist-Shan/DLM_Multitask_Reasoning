"""Scheduler tests use synthetic logits; none require a model or a GPU."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

from dlm_coupling.decoding import decode


MASK = 99
NUMERIC_IDS = {"0": 10, "1": 11, "2": 12, "3": 13}


def canvas(slots=None):
    positions = slots or {"A.x": 1, "B.x": 3}
    tokens = [80] * (max(positions.values()) + 2)
    for position in positions.values():
        tokens[position] = MASK
    return SimpleNamespace(
        token_ids=tokens,
        slot_positions=positions,
        value_token_ids={slot: dict(NUMERIC_IDS) for slot in positions},
        prompt_length=1,
        metadata={},
    )


class FakeBackend:
    """Predictions depend only on visible tokens and position, never gold."""

    mask_id = MASK

    def __init__(self, rule=None):
        self.calls = []
        self.rule = rule or (lambda tokens, position: (11, 0.8))

    def decode_token(self, token):
        return str(token - 10) if 10 <= token <= 13 else "<other>"

    def predict(self, token_ids, positions, value_token_ids):
        self.calls.append((list(token_ids), list(positions), deepcopy(value_token_ids)))
        results = []
        for position, requested in zip(positions, value_token_ids):
            token, confidence = self.rule(token_ids, position)
            results.append({
                "token_id": token,
                "token_text": self.decode_token(token),
                "confidence": confidence,
                "raw_top_id": token,
                "raw_top_text": self.decode_token(token),
                "probabilities": {str(item): confidence if item == token else 0.01 for item in requested},
            })
        return results


class DecodingTests(unittest.TestCase):
    def test_logs_predictions_before_writing_and_keeps_fixed_canvas(self):
        prepared = canvas()
        original = deepcopy(prepared)
        backend = FakeBackend(lambda tokens, pos: (11 if pos == 1 else 12, 0.7 if pos == 1 else 0.9))
        result = decode(backend, prepared, {"A.x": 1, "B.x": 2})
        first, second = result["steps"]
        self.assertEqual(result["nfe"], 2)
        self.assertEqual(first["masked_slots"], ["A.x", "B.x"])
        self.assertEqual(first["committed_before"], [])
        self.assertEqual(first["selected_slot"], "B.x")
        self.assertEqual(first["current_values"], {"A.x": 1, "B.x": 2})
        self.assertEqual((first["masks_before"], first["masks_after"]), (2, 1))
        self.assertEqual(first["candidates"]["A.x"]["gold_probability"], 0.7)
        self.assertEqual(second["masked_slots"], ["A.x"])
        self.assertEqual(second["committed_before"], ["B.x"])
        self.assertEqual(backend.calls[0][0][3], MASK)
        self.assertEqual(backend.calls[1][0][3], 12)
        self.assertEqual(result["final_correct"], {"A.x": True, "B.x": True})
        self.assertEqual(prepared.token_ids, original.token_ids)
        self.assertEqual(result["initial_token_ids"], original.token_ids)
        for index in (0, 2, 4):
            self.assertEqual(result["final_token_ids"][index], original.token_ids[index])

    def test_changed_gold_cannot_change_trajectory(self):
        prepared = canvas()
        first_backend, second_backend = FakeBackend(), FakeBackend()
        first = decode(first_backend, prepared, {"A.x": 1, "B.x": 1})
        second = decode(second_backend, prepared, {"A.x": 2, "B.x": 3})
        self.assertEqual(first_backend.calls, second_backend.calls)
        self.assertEqual(first["final_token_ids"], second["final_token_ids"])
        self.assertEqual([s["selected_slot"] for s in first["steps"]], [s["selected_slot"] for s in second["steps"]])
        self.assertNotEqual(first["final_correct"], second["final_correct"])
        self.assertNotEqual(first["steps"][0]["candidates"]["A.x"]["gold_probability"], second["steps"][0]["candidates"]["A.x"]["gold_probability"])

    def test_correct_to_wrong_prediction_after_partner_write_is_visible(self):
        def rule(tokens, position):
            if position == 3:
                return 12, 0.95
            return (11 if tokens[3] == MASK else 13), 0.6

        result = decode(FakeBackend(rule), canvas(), {"A.x": 1, "B.x": 2})
        self.assertTrue(result["steps"][0]["current_correct"]["A.x"])
        self.assertFalse(result["steps"][1]["current_correct"]["A.x"])
        self.assertEqual(result["steps"][0]["selected_slot"], "B.x")
        self.assertEqual(result["final_values"], {"A.x": 3, "B.x": 2})

    def test_ties_use_absolute_position_not_mapping_insertion_order(self):
        prepared = canvas({"B.x": 3, "A.x": 1})
        result = decode(FakeBackend(), prepared, {"A.x": 1, "B.x": 1})
        self.assertEqual(result["steps"][0]["selected_slot"], "A.x")

    def test_alternating_skips_exhausted_task_and_selects_confidence_within_task(self):
        prepared = canvas({"A.x": 1, "A.y": 3, "B.x": 5, "B.y": 7, "B.z": 9})
        scores = {1: 0.4, 3: 0.5, 5: 0.7, 7: 0.9, 9: 0.6}
        backend = FakeBackend(lambda tokens, pos: (11, scores[pos]))
        gold = {slot: 1 for slot in prepared.slot_positions}
        confidence = decode(backend, prepared, gold)
        alternating = decode(backend, prepared, gold, policy="alternating")
        start_b = decode(backend, prepared, gold, policy="alternating", starting_task="B")
        self.assertEqual([s["selected_slot"] for s in confidence["steps"]], ["B.y", "B.x", "B.z", "A.y", "A.x"])
        self.assertEqual([s["selected_slot"] for s in alternating["steps"]], ["A.y", "B.y", "A.x", "B.x", "B.z"])
        self.assertEqual([s["selected_slot"] for s in start_b["steps"]], ["B.y", "A.y", "B.x", "A.x", "B.z"])
        self.assertEqual(confidence["nfe"], alternating["nfe"])

    def test_forced_branch_is_visible_at_first_forward_and_never_overwritten(self):
        def rule(tokens, position):
            return (13 if tokens[3] == 12 else 11), 0.8

        prepared = canvas()
        backend = FakeBackend(rule)
        result = decode(backend, prepared, {"A.x": 1, "B.x": 1}, forced_values={"B.x": 2})
        self.assertEqual(result["forced_slots"], ["B.x"])
        self.assertEqual(result["nfe"], 1)
        self.assertEqual(backend.calls[0][0][3], 12)
        self.assertEqual(backend.calls[0][1], [1])
        self.assertEqual(result["initial_token_ids"][3], 12)
        self.assertEqual(result["final_token_ids"][3], 12)
        self.assertEqual(result["steps"][0]["committed_before"], ["B.x"])
        self.assertEqual(result["final_values"], {"A.x": 3, "B.x": 2})
        self.assertFalse(result["final_correct"]["B.x"])
        self.assertEqual(prepared.token_ids[3], MASK)

    def test_all_forced_slots_need_no_forward(self):
        backend = FakeBackend()
        result = decode(backend, canvas(), {"A.x": 1, "B.x": 2}, forced_values={"A.x": 1, "B.x": 2})
        self.assertEqual(result["nfe"], 0)
        self.assertEqual(result["steps"], [])
        self.assertEqual(backend.calls, [])

    def test_non_numeric_or_eos_candidate_is_invalid_value_but_does_not_stop_decode(self):
        result = decode(FakeBackend(lambda tokens, pos: (78, 0.9)), canvas(), {"A.x": 1, "B.x": 1})
        self.assertEqual(result["nfe"], 2)
        self.assertEqual(result["final_values"], {"A.x": None, "B.x": None})
        self.assertEqual(result["final_correct"], {"A.x": False, "B.x": False})
        self.assertEqual(result["steps"][0]["candidates"]["A.x"]["token_id"], 78)

    def test_backend_positions_are_used_without_decoder_logits_shift(self):
        prepared = canvas({"A.x": 2, "B.x": 4})
        backend = FakeBackend(lambda tokens, pos: (11 if pos == 2 else 12, 0.8))
        result = decode(backend, prepared, {"A.x": 1, "B.x": 2})
        self.assertEqual(backend.calls[0][1], [2, 4])
        self.assertEqual(result["final_values"], {"A.x": 1, "B.x": 2})

    def test_mask_candidate_is_rejected_even_when_it_was_raw_top(self):
        with self.assertRaisesRegex(ValueError, "other than mask_id"):
            decode(FakeBackend(lambda tokens, pos: (MASK, 0.9)), canvas(), {"A.x": 1, "B.x": 1})

    def test_raw_top_can_be_mask_if_selected_candidate_is_not(self):
        class MaskTopBackend(FakeBackend):
            def predict(self, *args):
                results = super().predict(*args)
                for result in results:
                    result.update(raw_top_id=MASK, raw_top_text="<mask>")
                return results

        result = decode(MaskTopBackend(), canvas(), {"A.x": 1, "B.x": 1})
        self.assertEqual(result["steps"][0]["candidates"]["A.x"]["raw_top_id"], MASK)
        self.assertNotIn(MASK, result["final_token_ids"])

    def test_rejects_bad_forced_keys_and_values(self):
        for forced in ({"C.x": 1}, {"B.x": 4}, {"B.x": "1"}, {"B.x": True}):
            with self.subTest(forced=forced), self.assertRaises(ValueError):
                decode(FakeBackend(), canvas(), {"A.x": 1, "B.x": 1}, forced_values=forced)

    def test_rejects_malformed_initial_canvas_and_gold(self):
        base = canvas()
        broken = []
        duplicate = deepcopy(base)
        duplicate.slot_positions["B.x"] = 1
        broken.append(duplicate)
        missing_mask = deepcopy(base)
        missing_mask.token_ids[1] = 11
        broken.append(missing_mask)
        extra_mask = deepcopy(base)
        extra_mask.token_ids[0] = MASK
        broken.append(extra_mask)
        out_of_range = deepcopy(base)
        out_of_range.slot_positions["B.x"] = 500
        broken.append(out_of_range)
        duplicate_value = deepcopy(base)
        duplicate_value.value_token_ids["A.x"]["2"] = 11
        broken.append(duplicate_value)
        for prepared in broken:
            with self.subTest(prepared=prepared), self.assertRaises(ValueError):
                decode(FakeBackend(), prepared, {"A.x": 1, "B.x": 1})
        for gold in ({"A.x": 1}, {"A.x": 1, "B.x": 4}):
            with self.subTest(gold=gold), self.assertRaises(ValueError):
                decode(FakeBackend(), base, gold)

    def test_rejects_invalid_policy_and_start(self):
        for options in ({"policy": "native"}, {"starting_task": "C"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                decode(FakeBackend(), canvas(), {"A.x": 1, "B.x": 1}, **options)

    def test_backend_incomplete_probabilities_or_invalid_confidence_fails_loudly(self):
        class MissingProbabilityBackend(FakeBackend):
            def predict(self, *args):
                results = super().predict(*args)
                results[0]["probabilities"].pop("10")
                return results

        with self.assertRaisesRegex(ValueError, "omitted"):
            decode(MissingProbabilityBackend(), canvas(), {"A.x": 1, "B.x": 1})
        for score in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(score=score), self.assertRaises(ValueError):
                decode(FakeBackend(lambda tokens, pos: (11, score)), canvas(), {"A.x": 1, "B.x": 1})


if __name__ == "__main__":
    unittest.main()
