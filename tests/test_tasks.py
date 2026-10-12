import json
import copy
import random
import unittest

from dlm_coupling.tasks import generate_families, render_prompt, validate_family


def evaluate_graph(graph):
    values = {}
    depths = {}
    expected_edges = []
    for node in graph["nodes"]:
        if node["op"] == "const":
            value, depth = node["value"], 0
        elif node["op"] == "add":
            left, right = node["inputs"]
            value = values[left] + values[right]
            depth = max(depths[left], depths[right]) + 1
        else:
            raise AssertionError(f"Unsupported operation: {node['op']}")
        assert value == node["value"], node
        values[node["id"]] = value
        depths[node["id"]] = depth
        expected_edges.extend([[source, node["id"]] for source in node["inputs"]])
    assert expected_edges == graph["edges"]
    return {slot: values[node] for slot, node in graph["outputs"].items()}, depths


class ArithmeticTaskTests(unittest.TestCase):
    def test_common_single_digit_domain_is_declared_and_validated(self):
        families = generate_families(2, 17, "colab_digit", max_value=9)
        self.assertEqual(families, generate_families(2, 17, "colab_digit", max_value=9))
        for family in families:
            validate_family(family)
            self.assertEqual(family["max_value"], 9)
            for variant in family["variants"]:
                self.assertEqual(variant["metadata"]["output_range"], [0, 9])
                self.assertTrue(all(0 <= x <= 9 for x in variant["gold"].values()))
        changed = copy.deepcopy(families[0])
        changed["max_value"] = 99
        with self.assertRaises(ValueError):
            validate_family(changed)
        for invalid in (8, 100, True):
            with self.assertRaises(ValueError):
                generate_families(1, 17, max_value=invalid)

    def test_reproducible_local_rng_and_json(self):
        random.seed(83)
        before = random.getstate()
        families = generate_families(8, seed=17)
        self.assertEqual(before, random.getstate())
        self.assertEqual(families, generate_families(8, seed=17))
        self.assertNotEqual(families, generate_families(8, seed=23))
        self.assertEqual(families, json.loads(json.dumps(families)))
        self.assertNotEqual(families[0]["variants"][0]["metadata"]["semantic_id"],
                            generate_families(1, 17, split="pilot")[0]["variants"][0]["metadata"]["semantic_id"])

    def test_graph_truth_depth_and_counterfactual_labels(self):
        for depth in (2, 3, 4, 12):
            for family in generate_families(16, seed=41, depth=depth):
                self.assertEqual(6, len(family["variants"]))
                lookup = {(v["condition"], v["perturbation"]): v for v in family["variants"]}
                for variant in family["variants"]:
                    with self.subTest(depth=depth, variant=variant["variant_id"]):
                        truth, depths = evaluate_graph(variant["graph"])
                        self.assertEqual(truth, variant["gold"])
                        for question in ("A", "B"):
                            self.assertEqual(depth, depths[variant["graph"]["outputs"][f"{question}.answer"]])
                            self.assertEqual(depth - 1, depths[variant["graph"]["outputs"][f"{question}.intermediate"]])
                        self.assertTrue(all(0 <= value <= 99 for value in truth.values()))
                        base = lookup[(variant["condition"], "base")]
                        actual_affected = [q for q in ("A", "B") if truth[f"{q}.answer"] != base["gold"][f"{q}.answer"]]
                        self.assertEqual(actual_affected, variant["affected_questions"])
                        self.assertEqual(base["questions"], variant["questions"])
                        for slot, value in truth.items():
                            expected_delta = 2 if slot.split(".")[0] in actual_affected else 0
                            self.assertEqual(expected_delta, value - base["gold"][slot])

    def test_conditions_change_only_question_dependency(self):
        for family in generate_families(10, 17):
            lookup = {(v["condition"], v["perturbation"]): v for v in family["variants"]}
            for perturbation in ("base", "red", "blue"):
                independent = lookup[("independent", perturbation)]
                shared = lookup[("shared", perturbation)]
                self.assertEqual(independent["facts"], shared["facts"])
                self.assertEqual(independent["questions"]["A"], shared["questions"]["A"])
                self.assertEqual(independent["questions"]["B"].replace("blue", "red"), shared["questions"]["B"])
                self.assertEqual(shared["gold"]["A.intermediate"], shared["gold"]["B.intermediate"])
                self.assertEqual(4, len(set(independent["gold"].values())))
                self.assertNotEqual(shared["gold"]["A.answer"], shared["gold"]["B.answer"])
                self.assertIn("blue.receipt0", {node["id"] for node in shared["graph"]["nodes"]})

    def test_unique_semantic_records_and_ids(self):
        families = generate_families(256, 17, split="pilot", depth=2)
        self.assertEqual(256, len({f["family_id"] for f in families}))
        self.assertEqual(256, len({f["variants"][0]["metadata"]["semantic_id"] for f in families}))
        self.assertEqual(1536, len({v["variant_id"] for f in families for v in f["variants"]}))

    def test_validator_accepts_generated_and_rejects_corruption(self):
        families = generate_families(12, 23, depth=3)
        for family in families:
            self.assertIsNone(validate_family(family))
        original = families[0]
        corruptions = [
            lambda f: f["variants"][0]["gold"].__setitem__("A.answer", 99),
            lambda f: f["variants"][1].__setitem__("affected_questions", []),
            lambda f: f["variants"][0]["graph"]["nodes"][0].__setitem__("value", 99),
            lambda f: f["variants"][0]["metadata"].__setitem__("split", "pilot"),
            lambda f: f["variants"][0]["facts"].__setitem__(0, "Red station received 999 boxes."),
            lambda f: f["variants"].__setitem__(1, f["variants"][0]),
            lambda f: f.__setitem__("depth", 4),
        ]
        for corrupt in corruptions:
            family = copy.deepcopy(original)
            corrupt(family)
            with self.assertRaises(ValueError):
                validate_family(family)
        with self.assertRaises(ValueError):
            validate_family({})

    def test_render_layout_and_no_gold_dependency(self):
        variant = generate_families(1, 17)[0]["variants"][0]
        for input_order in ("AB", "BA"):
            for output_order in ("AB", "BA"):
                for format_name in ("answer_only", "with_intermediate"):
                    prompt = render_prompt(variant, input_order, output_order, format_name)
                    first_source, second_source = ("Red station", "Blue station") if input_order == "AB" else ("Blue station", "Red station")
                    self.assertLess(prompt.index(first_source), prompt.index(second_source))
                    self.assertLess(prompt.index(variant["questions"][input_order[0]]), prompt.index(variant["questions"][input_order[1]]))
                    expected = [f"{q}: answer=<integer>" if format_name == "answer_only" else f"{q}: intermediate=<integer>; answer=<integer>" for q in output_order]
                    self.assertEqual(expected, prompt.splitlines()[-2:])
                    # Changing hidden gold cannot alter a prompt. A number
                    # may occur both in the facts and the gold by chance.
                    poisoned = dict(variant, gold={slot: "LEAK_SENTINEL" for slot in variant["gold"]})
                    self.assertEqual(prompt, render_prompt(poisoned, input_order, output_order, format_name))

    def test_invalid_arguments(self):
        for kwargs in ({"count": -1}, {"count": True}, {"seed": "17"}, {"depth": 1}, {"depth": 13}, {"split": ""}, {"split": "dev/pilot"}):
            args = {"count": 1, "seed": 17}
            args.update(kwargs)
            with self.assertRaises(ValueError):
                generate_families(**args)
        self.assertEqual([], generate_families(0, 17))
        variant = generate_families(1, 17)[0]["variants"][0]
        for kwargs in ({"input_order": "AA"}, {"output_order": "A"}, {"format_name": "verbose"}):
            with self.assertRaises(ValueError):
                render_prompt(variant, **kwargs)


if __name__ == "__main__":
    unittest.main()
