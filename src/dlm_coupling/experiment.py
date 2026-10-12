"""One design shared by every model; prepare and audit before GPU inference."""
from __future__ import annotations

from .storage import digest
from .tasks.arithmetic import render_prompt, validate_family


def slot_keys(format_name, output_order):
    if format_name not in {"answer_only", "with_intermediate"}:
        raise ValueError(f"Unknown output format: {format_name}")
    if output_order not in {"AB", "BA"}:
        raise ValueError("Output order must be AB or BA")
    fields = ["answer"] if format_name == "answer_only" else ["intermediate", "answer"]
    return [f"{task}.{field}" for task in output_order for field in fields]


def validate_config(config):
    if not config.get("formats") or len(set(config["formats"])) != len(config["formats"]):
        raise ValueError("formats must be a nonempty unique list")
    if not config.get("layouts") or len(set(config["layouts"])) != len(config["layouts"]):
        raise ValueError("layouts must be a nonempty unique list")
    for form in config["formats"]:
        slot_keys(form, "AB")
    for layout in config["layouts"]:
        if layout not in {"AB_AB", "AB_BA", "BA_AB", "BA_BA"}:
            raise ValueError(f"Invalid layout {layout}")
    if config.get("policy", "confidence") not in {"confidence", "alternating"}:
        raise ValueError("policy must be confidence or alternating")
    if config.get("starting_task", "A") not in {"A", "B"}:
        raise ValueError("starting_task must be A or B")
    if int(config.get("bootstrap_samples", 2000)) < 1:
        raise ValueError("bootstrap_samples must be positive")


def validate_dataset(families):
    if not families:
        raise ValueError("Dataset is empty")
    ids, semantics = set(), set()
    for family in families:
        validate_family(family)
        key = family["family_id"]
        if key in ids:
            raise ValueError(f"Duplicate family ID: {key}")
        ids.add(key)
        semantic = family["variants"][0]["metadata"].get("semantic_id")
        if semantic:
            if semantic in semantics:
                raise ValueError("Duplicate semantic family; check split overlap")
            semantics.add(semantic)


def cases(families, config):
    """Yield complete paired families; limits must never cut a family in half."""
    validate_config(config)
    for family in families:
        variants = {(v["condition"], v["perturbation"]): v for v in family["variants"]}
        for variant in family["variants"]:
            for form in config["formats"]:
                for layout in config["layouts"]:
                    inp, out = layout.split("_")
                    slots = slot_keys(form, out)
                    base = {
                        "family_id": family["family_id"], "variant_id": variant["variant_id"],
                        "condition": variant["condition"], "perturbation": variant["perturbation"],
                        "format": form, "layout": layout,
                        "prompt": render_prompt(variant, inp, out, form), "slots": slots,
                        "gold": {key: variant["gold"][key] for key in slots},
                        "affected_questions": variant["affected_questions"],
                        "branch": "baseline", "forced_values": {},
                    }
                    base["run_key"] = digest({k: v for k, v in base.items() if k not in {"prompt", "gold"}})
                    yield base
                    # Output interventions are predeclared only at the base prompt.
                    # The alternate tuple comes from the factual counterfactual
                    # that changes B's true answer in this dependency condition.
                    if config.get("interventions", False) and variant["perturbation"] == "base":
                        donor_source = "blue" if variant["condition"] == "independent" else "red"
                        donor = variants[(variant["condition"], donor_source)]
                        for branch, values in [("b_correct", variant["gold"]), ("b_alternate", donor["gold"])]:
                            item = dict(base)
                            item["branch"] = branch
                            item["forced_values"] = {k: values[k] for k in slots if k.startswith("B.")}
                            item["donor_source"] = donor_source
                            item["run_key"] = digest({k: v for k, v in item.items() if k not in {"prompt", "gold", "run_key"}})
                            yield item


def audit(backend, families, config):
    """Refuse incompatible pairs; do not quietly select a model-specific dataset."""
    validate_dataset(families)
    prepared = {}
    group_lengths = {}
    domain = set(range(100))
    required_values = {f["family_id"]: {value for v in f["variants"] for value in v["gold"].values()} for f in families}
    count = 0
    for case in cases(families, config):
        key = (case["prompt"], tuple(case["slots"]))
        if key not in prepared:
            prepared[key] = backend.prepare(*key)
        canvas = prepared[key]
        for slot in case["slots"]:
            mapping = canvas.value_token_ids[slot]
            domain.intersection_update(int(v) for v in mapping)
            for value in required_values[case["family_id"]]:
                if str(value) not in mapping:
                    raise ValueError(f"{case['family_id']} {slot} value {value} is not one token with stable boundaries. Regenerate a common dataset or redesign the interface; do not skip this case.")
        group = (case["family_id"], case["condition"], case["format"], case["layout"])
        shape = (canvas.prompt_length, len(canvas.token_ids), tuple(canvas.slot_positions.items()))
        if group in group_lengths and group_lengths[group] != shape:
            raise ValueError(f"Counterfactual token length/slot position mismatch: {group}; reject the complete family for BOTH models before evaluating")
        group_lengths[group] = shape
        count += 1
    return {
        "status": "passed", "family_count": len(families), "case_count": count,
        "common_slot_integer_domain": sorted(domain), "dataset_digest": digest(families),
        "config_digest": digest(config), "backend": backend.metadata,
        "scope": "tokenization and paired shapes; no model forwards or accuracy validation",
    }, prepared
