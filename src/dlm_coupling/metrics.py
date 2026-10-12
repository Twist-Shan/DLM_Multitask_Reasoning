"""Paired readouts with the semantic family as the bootstrap unit."""
from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path
import random
from statistics import mean

from .storage import write_json


def cluster_interval(observations, seed=123, samples=2000):
    """Each observation is (family_id, value); repeated variants are averaged."""
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    groups = defaultdict(list)
    for family, value in observations:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Non-finite metric value")
        groups[family].append(value)
    values = [mean(groups[key]) for key in sorted(groups)]
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_families": 0, "n_observations": 0}
    estimate = mean(values)
    low = high = None
    if len(values) > 1:
        rng = random.Random(seed)
        draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
        low = draws[int(0.025 * (samples - 1))]
        high = draws[int(0.975 * (samples - 1))]
    return {"mean": estimate, "ci_low": low, "ci_high": high,
            "n_families": len(values), "n_observations": sum(map(len, groups.values()))}


def summarize_records(records, seed=123, samples=2000):
    buckets = defaultdict(list)
    seen = set()
    groups = defaultdict(set)
    for record in records:
        group = tuple(record[key] for key in ("family_id", "condition", "format", "layout"))
        groups[group].add((record["perturbation"], record["branch"]))
    required = {(p, "baseline") for p in ("base", "red", "blue")}
    interventions = {("base", "b_correct"), ("base", "b_alternate")}
    for group, present in groups.items():
        if not required <= present or present - required not in (set(), interventions):
            raise ValueError(f"Incomplete or invalid paired conditions for {group}; finish/resume the complete family")

    def add(record, metric, value, contrast="behavior"):
        # Save both all-layout and per-layout readouts. Repeats are clustered.
        for layout in ("all", record["layout"]):
            key = (record["condition"], record["format"], layout, contrast, metric)
            buckets[key].append((record["family_id"], float(value)))

    index = {}
    for record in records:
        if record["run_key"] in seen:
            raise ValueError("Duplicate run_key in results; do not count resumed rows twice")
        seen.add(record["run_key"])
        trace = record["trace"]
        key = (record["family_id"], record["condition"], record["format"], record["layout"], record["perturbation"], record["branch"])
        if key in index:
            raise ValueError("Duplicate experimental condition")
        index[key] = record
        if record["branch"] == "baseline":
            correct = trace["final_correct"]
            for task in ("A", "B"):
                slot = f"{task}.answer"
                add(record, f"{task}_accuracy", correct[slot])
                add(record, f"{task}_invalid", trace["final_values"][slot] is None)
            add(record, "both_answers_correct", correct["A.answer"] and correct["B.answer"])
            add(record, "all_requested_values_correct", all(correct.values()))
            a, b = record["gold"]["A.answer"], record["gold"]["B.answer"]
            swapped = a != b and trace["final_values"]["A.answer"] == b and trace["final_values"]["B.answer"] == a
            add(record, "complete_answer_swap_unconditional", swapped)
            add(record, "answer_distinct_coverage", a != b)
            add(record, "first_forward_both_answers_correct", all(trace["steps"][0]["current_correct"][f"{q}.answer"] for q in ("A", "B")))

    for key, base in index.items():
        family, condition, form, layout, perturbation, branch = key
        if perturbation != "base" or branch != "baseline":
            continue
        for source in ("red", "blue"):
            changed = index.get((family, condition, form, layout, source, "baseline"))
            if changed is None:
                raise ValueError("Incomplete counterfactual family; finish/resume the run before summarizing")
            for task in ("A", "B"):
                slot = f"{task}.answer"
                affected = task in changed["affected_questions"]
                category = "affected" if affected else "unaffected"
                contrast = "input_" + source
                add(base, category + "_joint_correct", base["trace"]["final_correct"][slot] and changed["trace"]["final_correct"][slot], contrast)
                bp = base["trace"]["steps"][0]["candidates"][slot]
                cp = changed["trace"]["steps"][0]["candidates"][slot]
                add(base, category + "_first_candidate_flip", bp["token_id"] != cp["token_id"], contrast)
                old, new = str(base["gold"][slot]), str(changed["gold"][slot])
                if affected:
                    # A directional paired contrast uses the SAME two values in both states.
                    required = {old, new}
                    if not (required <= bp["value_probabilities"].keys() and required <= cp["value_probabilities"].keys()):
                        raise ValueError("Missing paired probability probes; repeat the shared audit and run")
                    shift = (cp["value_probabilities"][new] - bp["value_probabilities"][new]
                             + bp["value_probabilities"][old] - cp["value_probabilities"][old]) / 2
                    add(base, "affected_directional_probability_shift", shift, contrast)
                else:
                    add(base, "unaffected_gold_probability_sensitivity", abs(cp["gold_probability"] - bp["gold_probability"]), contrast)
        correct = index.get((family, condition, form, layout, "base", "b_correct"))
        alternate = index.get((family, condition, form, layout, "base", "b_alternate"))
        if bool(correct) != bool(alternate):
            raise ValueError("Incomplete output-intervention pair")
        if correct is not None:
            if correct["trace"]["nfe"] != alternate["trace"]["nfe"]:
                raise ValueError("Filled-B branches must have identical forward budgets")
            slot = "A.answer"
            c, a = correct["trace"], alternate["trace"]
            if any(not key.startswith("B.") for key in c["forced_slots"] + a["forced_slots"]):
                raise ValueError("Output intervention must force only B slots")
            if slot not in c["steps"][0]["candidates"] or slot not in a["steps"][0]["candidates"]:
                raise ValueError("A.answer must remain masked at the first intervention forward")
            add(base, "A_accuracy_correct_minus_alternate", int(c["final_correct"][slot]) - int(a["final_correct"][slot]), "output_B")
            cp, ap = c["steps"][0]["candidates"][slot], a["steps"][0]["candidates"][slot]
            add(base, "A_gold_probability_correct_minus_alternate", cp["gold_probability"] - ap["gold_probability"], "output_B")
            inserted = str(alternate["forced_values"]["B.answer"])
            if inserted not in cp["value_probabilities"] or inserted not in ap["value_probabilities"]:
                raise ValueError("Missing intervention probability probe; repeat the shared audit and run")
            eligible = inserted != str(base["gold"][slot])
            add(base, "alternate_B_value_distinct_coverage", eligible, "output_B")
            if eligible:
                add(base, "A_alternate_B_probability_shift", ap["value_probabilities"][inserted] - cp["value_probabilities"][inserted], "output_B")
    result = []
    for key, observations in sorted(buckets.items()):
        condition, form, layout, contrast, metric = key
        result.append({"condition": condition, "format": form, "layout": layout,
                       "contrast": contrast, "metric": metric,
                       **cluster_interval(observations, seed, samples)})
    return result


def save_summary(output, records, manifest):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Report directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = manifest["experiment_config"]
    metrics = summarize_records(records, int(config.get("bootstrap_seed", 123)), int(config.get("bootstrap_samples", 2000)))
    result = {"status": "complete", "is_mock": manifest["is_mock"],
              "interpretation": "Mechanical test only; not model evidence" if manifest["is_mock"] else "Exploratory fixed-slot model measurements",
              "run_fingerprint": manifest["fingerprint"], "n_records": len(records),
              "statistical_unit": "family_id; layouts and factual variants are repeated measures",
              "interval": "95% percentile bootstrap over equally weighted family means; null CI for one family",
              "metrics": metrics}
    write_json(output / "summary.json", result)
    if metrics:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    return result
