"""Mechanical checks of the saved-probability carry diagnostic (no weights)."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_precommit_v22_carry import eligible, summarize, tens_effect


class CarryDiagnosticTests(unittest.TestCase):
    def test_decimal_boundary_and_borrow_not_accuracy(self):
        self.assertTrue(eligible(18, 29, 1))
        self.assertTrue(eligible(23, 30, -1))
        self.assertFalse(eligible(19, 29, 1))
        self.assertFalse(eligible(20, 30, -1))
        self.assertFalse(eligible(18, 28, 1))
        self.assertFalse(eligible(18, 99, 1))
        with self.assertRaises(ValueError):
            eligible(18, 29, 0)

    def test_tens_metric_uses_position_map_and_no_division_by_two(self):
        context = dict(prediction_positions=[6, 3, 7, 2],
                       Target={"spans": {"B": [6, 7]}},
                       B_target_token_ids=[12, 19], B_candidate_token_ids=[13, 10])
        def prediction(target, candidate):
            return dict(token_id=12, probabilities={"12": target, "13": candidate})
        context["baseline"] = [prediction(.4, .1), {}, {}, {}]
        patch = {"predictions": [prediction(.2, .2), {}, {}, {}]}
        self.assertAlmostEqual(tens_effect(context, patch)["delta_tens"], math.log(4))
        patch["predictions"][0]["probabilities"]["13"] = 0
        with self.assertRaises(ValueError):
            tens_effect(context, patch)

    def test_groups_not_repeated_directions_define_mean(self):
        result = summarize([dict(case="a", delta_tens=1), dict(case="a", delta_tens=3),
                            dict(case="b", delta_tens=-1)])
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["directions"], 3)
        self.assertEqual(result["mean_tens_delta"], .5)
        self.assertEqual(result["positive_groups"], 1)
        self.assertIsNone(summarize([])["mean_tens_delta"])
        rounding = summarize([dict(case="x", delta_tens=4.35e-9)])
        self.assertEqual(rounding["strict_positive_groups"], 1)
        self.assertEqual(rounding["positive_groups"], 0)
        self.assertEqual(rounding["zero_groups"], 1)


if __name__ == "__main__":
    unittest.main()
