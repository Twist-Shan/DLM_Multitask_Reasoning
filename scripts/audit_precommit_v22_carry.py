"""Post-hoc carry/borrow diagnostic from saved probabilities; no model inference.

Select by arithmetic alone: B +/- 1 crosses a decimal boundary but A +/- 1
does not. Report the B tens position, without averaging it with the ones
position. Calibration and discovery remain separately visible; pooling them
does not turn this newly selected diagnostic into held-out confirmation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dlm_coupling import precommit_v2 as v2

ZERO_TOLERANCE = 1e-6

def eligible(a, b, sign):
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    return (all(10 <= n <= 99 for n in (a, b, a + sign, b + sign))
            and a // 10 == (a + sign) // 10
            and b // 10 != (b + sign) // 10)


def tens_effect(context, patch):
    """Recompute the one-position log-ratio change, with explicit alignment."""
    positions = context["prediction_positions"]
    b_positions = context["Target"]["spans"]["B"]
    if (len(set(positions)) != len(positions) or len(b_positions) != 2
            or len(context["B_target_token_ids"]) != 2
            or len(context["B_candidate_token_ids"]) != 2):
        raise ValueError("Expected unique positions and two audited digit tokens")
    target_id = str(context["B_target_token_ids"][0])
    candidate_id = str(context["B_candidate_token_ids"][0])
    if target_id == candidate_id:
        raise ValueError("A carry/borrow must change B's tens token")
    terms = {}
    for label, predictions in (("before", context["baseline"]),
                               ("after", patch["predictions"])):
        if len(positions) != len(predictions):
            raise ValueError("Prediction positions and probabilities do not align")
        pred = dict(zip(positions, predictions))[b_positions[0]]
        probs = pred["probabilities"]
        target, candidate = probs[target_id], probs[candidate_id]
        # Do not silently floor, impute or renormalize the saved probabilities.
        if not all(math.isfinite(x) and 0 < x <= 1 for x in (target, candidate)):
            raise ValueError("Missing, zero or invalid saved probability")
        terms[label] = {"target_probability": target,
                        "candidate_probability": candidate,
                        "log_ratio": math.log(candidate) - math.log(target),
                        "argmax_is_candidate": str(pred["token_id"]) == candidate_id}
    return {"delta_tens": terms["after"]["log_ratio"] - terms["before"]["log_ratio"],
            **terms}


def summarize(rows):
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r["delta_tens"])
    values = [statistics.mean(v) for v in by_case.values()]
    return {"groups": len(values), "directions": len(rows),
            "mean_tens_delta": statistics.mean(values) if values else None,
            "positive_groups": sum(v > ZERO_TOLERANCE for v in values),
            "zero_groups": sum(abs(v) <= ZERO_TOLERANCE for v in values),
            "negative_groups": sum(v < -ZERO_TOLERANCE for v in values),
            "strict_positive_groups": sum(v > 0 for v in values),
            "positive_directions": sum(r["delta_tens"] > ZERO_TOLERANCE for r in rows),
            "rows": rows}


def audit(runs):
    all_rows, hashes, eligible_by_split = [], {}, {}
    for split in ("calibration", "discovery"):
        folder = runs / "illada-v22" / split
        files = {name: folder / name for name in ("contexts.jsonl", "patch.jsonl")}
        parsed = {}
        for name, path in files.items():
            raw = path.read_bytes()
            hashes[f"illada-v22/{split}/{name}"] = hashlib.sha256(raw).hexdigest()
            parsed[name] = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        contexts = {}
        selectors = defaultdict(dict)
        for c in parsed["contexts.jsonl"]:
            if c["context_id"] in contexts:
                raise ValueError("Duplicate context_id")
            contexts[c["context_id"]] = c
            if c["source_kind"] != "change_A":
                continue
            expected = v2.gold(c["p"], c["relation"])
            if c["expected"] != expected:
                raise ValueError("Target arithmetic differs from recorded truth")
            sign = c["sign"]
            if eligible(expected["A"], expected["B"], sign):
                selectors[c["relation"]][(c["case"], sign)] = dict(
                    case=c["case"], sign=sign, A=expected["A"], B=expected["B"],
                    A_candidate=expected["A"] + sign, B_candidate=expected["B"] + sign)
        eligible_by_split[split] = {rel: list(rows.values()) for rel, rows in selectors.items()}
        seen = set()
        for p in parsed["patch.jsonl"]:
            if p["mode"] == "resample":
                continue
            c = contexts[p["context_id"]]
            if (c["case"], c["sign"]) not in selectors[c["relation"]]:
                continue
            for key in ("case", "relation", "source_kind", "sign"):
                if p[key] != c[key]:
                    raise ValueError(f"Context/patch mismatch: {key}")
            if c["alternative"]["B"] != c["expected"]["B"] + c["sign"]:
                raise ValueError("Candidate must be Target B + sign, including controls")
            if c["source_kind"] in ("change_A", "A_change_B_fixed"):
                a = c["expected"]["A"]
                if c["source_expected"]["A"] != a + c["sign"]:
                    raise ValueError("Source A did not change by sign")
                if a // 10 != c["source_expected"]["A"] // 10:
                    raise ValueError("Source A tens digit changed")
            condition = "/".join((c["relation"], c["source_kind"], p["mode"],
                                   "-".join(map(str, p["layers"])), p["site"]))
            identity = (condition, c["case"], c["sign"])
            if identity in seen:
                raise ValueError("Duplicate case/direction within a condition")
            seen.add(identity)
            all_rows.append(dict(split=split, condition=condition, case=c["case"],
                                 sign=c["sign"], Target=c["expected"],
                                 Source=c["source_expected"], candidate=c["alternative"],
                                 **tens_effect(c, p)))
    summaries = {}
    for split in ("calibration", "discovery", "pooled"):
        by_condition = defaultdict(list)
        for r in all_rows:
            if split == "pooled" or r["split"] == split:
                by_condition[r["condition"]].append(r)
        summaries[split] = {k: summarize(v) for k, v in sorted(by_condition.items())}
    return dict(analysis="v2.2 post-hoc carry/borrow", inference_performed=False,
                selection="Target B+sign changes tens; Target A+sign retains tens; no accuracy filter",
                control_selection="Reuse change_A-eligible case/sign pairs within each relation",
                metric="B tens-only log[P(candidate)/P(Target)] after minus before, in nat",
                aggregation="Mean directions within case, then mean cases",
                zero_tolerance=ZERO_TOLERANCE,
                sign_count_note="Both strict >0 and >1e-6 counts saved; tiny softmax roundoff is not a positive effect",
                interpretation="Small post-hoc subset; no claim of an addition algorithm or carry circuit",
                input_sha256=hashes, eligible=eligible_by_split, summaries=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="Directory containing illada-v22")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit(args.runs)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    for k, v in result["summaries"]["pooled"].items():
        if k.endswith("/answer_A") and ("/single/30/" in k or "/single/31/" in k or "/sustained/24-" in k):
            print(k, {s: v[s] for s in ("groups", "directions", "mean_tens_delta", "positive_groups")})


if __name__ == "__main__":
    main()
