import copy
import unittest

from dlm_coupling.backends import MockBackend, make_backend
from dlm_coupling.experiment import audit, cases, validate_dataset
from dlm_coupling.tasks.arithmetic import generate_families


CONFIG = {"formats": ["answer_only", "with_intermediate"], "layouts": ["AB_AB"], "interventions": True}


class ExperimentTests(unittest.TestCase):
    def test_design_and_interventions_have_matched_budgets(self):
        data = generate_families(1, 17)
        design = list(cases(data, CONFIG))
        self.assertEqual(len(design), 20)
        self.assertEqual(len({c["run_key"] for c in design}), len(design))
        self.assertEqual(sum(len(c["slots"]) - len(c["forced_values"]) for c in design), 48)
        for case in design:
            if case["branch"] != "baseline":
                self.assertEqual(case["perturbation"], "base")
                self.assertTrue(all(k.startswith("B.") for k in case["forced_values"]))
                if case["branch"] == "b_alternate":
                    self.assertNotEqual(case["forced_values"]["B.answer"], case["gold"]["B.answer"])

    def test_audit_checks_all_counterfactual_values_and_shapes(self):
        data = generate_families(1, 17)
        report, cache = audit(MockBackend(), data, CONFIG)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["common_slot_integer_domain"], list(range(100)))
        self.assertEqual(len(cache), 12)

        class MissingValueBackend(MockBackend):
            def prepare(self, prompt, keys):
                canvas = super().prepare(prompt, keys)
                missing = str(data[0]["variants"][-1]["gold"]["A.answer"])
                for mapping in canvas.value_token_ids.values():
                    mapping.pop(missing, None)
                return canvas
        with self.assertRaisesRegex(ValueError, "not one token"):
            audit(MissingValueBackend(), data, CONFIG)

    def test_semantic_duplicates_with_different_ids_are_rejected(self):
        # Relabel a valid family without changing its semantic arithmetic.
        first = generate_families(1, 17, "dev")
        second = copy.deepcopy(first)
        previous = second[0]["family_id"]
        renamed = previous.rsplit(".", 1)[0] + ".99999"
        second[0]["family_id"] = renamed
        for variant in second[0]["variants"]:
            variant["variant_id"] = variant["variant_id"].replace(previous, renamed)
            variant["metadata"]["family_id"] = renamed
        with self.assertRaisesRegex(ValueError, "Duplicate semantic"):
            validate_dataset(first + second)

    def test_factory_requires_explicit_backend_and_accepts_named_fixture(self):
        with self.assertRaises(ValueError):
            make_backend({"family": "dream"})
        self.assertTrue(make_backend({"backend": "mock", "family": "mock", "model_id": "named-fixture"}).metadata["is_mock"])


if __name__ == "__main__":
    unittest.main()
