import copy
import unittest

from dlm_coupling.experiment import cases
from dlm_coupling.metrics import cluster_interval, summarize_records
from dlm_coupling.tasks.arithmetic import generate_families


def perfect_records():
    """Hand-constructed oracle RECORDS for scoring tests; not an inference backend."""
    config = {"formats": ["answer_only"], "layouts": ["AB_AB"], "interventions": True}
    records = []
    for case in cases(generate_families(2, 17), config):
        values = dict(case["gold"])
        values.update(case["forced_values"])
        candidates = {}
        for slot in case["slots"]:
            if slot in case["forced_values"]:
                continue
            probabilities = {str(i): 0.2 / 99 for i in range(100)}
            probabilities[str(case["gold"][slot])] = 0.8
            candidates[slot] = {"token_id": values[slot] + 1, "gold_probability": 0.8, "value_probabilities": probabilities}
        records.append({**case, "trace": {
            "nfe": len(case["slots"]) - len(case["forced_values"]),
            "final_values": values, "final_correct": {k: values[k] == case["gold"][k] for k in values},
            "forced_slots": list(case["forced_values"]),
            "steps": [{"candidates": candidates, "current_correct": {k: values[k] == case["gold"][k] for k in values}}],
        }})
    return records


class MetricsTests(unittest.TestCase):
    def test_family_means_have_equal_weight(self):
        result = cluster_interval([("A", 1)] * 100 + [("B", 0)], samples=200)
        self.assertEqual(result["mean"], 0.5)
        self.assertEqual(result["n_families"], 2)
        self.assertEqual(result["n_observations"], 101)
        self.assertEqual(result, cluster_interval([("A", 1)] * 100 + [("B", 0)], samples=200))
        self.assertIsNone(cluster_interval([("A", 1)])["ci_low"])

    def test_correct_update_and_preservation_with_known_truth(self):
        rows = summarize_records(perfect_records(), samples=100)
        for row in rows:
            if row["metric"] in {"affected_joint_correct", "unaffected_joint_correct", "A_accuracy", "B_accuracy"}:
                self.assertEqual(row["mean"], 1.0)
            if row["metric"] in {"A_accuracy_correct_minus_alternate", "unaffected_gold_probability_sensitivity"}:
                self.assertEqual(row["mean"], 0.0)
        shifts = [r["mean"] for r in rows if r["metric"] == "affected_directional_probability_shift"]
        self.assertTrue(shifts and all(v > 0.79 for v in shifts))

    def test_forced_b_errors_are_not_counted_as_model_errors(self):
        rows = summarize_records(perfect_records(), samples=50)
        self.assertTrue(all(r["mean"] == 1 for r in rows if r["metric"] == "B_accuracy"))
        self.assertFalse(any(r["contrast"] == "output_B" and r["metric"] == "B_accuracy" for r in rows))

    def test_missing_base_or_orphan_interventions_fail(self):
        records = perfect_records()
        broken = [r for r in records if r["run_key"] != records[0]["run_key"]]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize_records(broken)
        orphan = [r for r in records if r["branch"] != "b_correct"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize_records(orphan)

    def test_duplicate_rows_and_mismatched_budgets_fail(self):
        records = perfect_records()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_records(records + [records[0]])
        changed = copy.deepcopy(records)
        next(r for r in changed if r["branch"] == "b_alternate")["trace"]["nfe"] = 9
        with self.assertRaisesRegex(ValueError, "identical forward budgets"):
            summarize_records(changed)

    def test_missing_cross_state_probe_cannot_silently_drop_observations(self):
        records = perfect_records()
        changed = next(r for r in records if r["perturbation"] == "red" and r["branch"] == "baseline")
        original = next(r for r in records if r["family_id"] == changed["family_id"] and r["condition"] == changed["condition"] and r["perturbation"] == "base" and r["branch"] == "baseline")
        changed["trace"]["steps"][0]["candidates"]["A.answer"]["value_probabilities"].pop(str(original["gold"]["A.answer"]))
        with self.assertRaisesRegex(ValueError, "Missing paired probability probes"):
            summarize_records(records)


if __name__ == "__main__":
    unittest.main()
