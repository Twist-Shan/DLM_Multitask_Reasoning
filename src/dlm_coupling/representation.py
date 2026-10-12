"""Explicit two-digit fields, with semantic scoring separate from token traces.

This is a controlled decimal representation, not native variable-length text.
All values reserve two positions without consulting gold or its digit length.
"""
from statistics import mean

from .backends.base import PreparedCanvas, output_template
from .decoding import decode


def digit_keys(slot):
    return [slot + "_tens", slot + "_ones"]


def expand_values(values):
    result = {}
    for slot, value in values.items():
        if type(value) is not int or not 0 <= value <= 99:
            raise ValueError("decimal_digits requires integer values in 0..99")
        result.update(zip(digit_keys(slot), (value // 10, value % 10)))
    return result


def digit_prompt(prompt, slots):
    prefix = prompt.split("Return only the following format,", 1)[0]
    keys = [key for slot in slots for key in digit_keys(slot)]
    return (prefix + "Represent every requested value by its tens and ones digits, including a zero tens digit "
            "for values below ten. For example, 7 is tens=0, ones=7; 27 is tens=2, ones=7. "
            "Each field must contain exactly one decimal digit.\nReturn only the following format:\n"
            + output_template(keys, ["<digit>"] * len(keys)))


def prepare_case(backend, case):
    if case.get("representation", "integer_token") == "integer_token":
        return backend.prepare(case["prompt"], case["slots"])
    keys = [key for slot in case["slots"] for key in digit_keys(slot)]
    original = backend.prepare(case["prompt"], keys)
    mappings = {}
    for key in keys:
        if not all(str(d) in original.value_token_ids[key] for d in range(10)):
            raise ValueError(f"All ten decimal digits must have stable single-token encodings at {key}")
        mappings[key] = {str(d): original.value_token_ids[key][str(d)] for d in range(10)}
    return PreparedCanvas(original.token_ids, original.slot_positions, mappings, original.prompt_length,
                          {**original.metadata, "representation": "decimal_digits", "numeric_width": 2,
                           "candidate_domain": "each field: digits 0..9; semantic values 0..99",
                           "semantic_slots": case["slots"],
                           "score_kind": "mean_digit_probability; not an answer probability"})


def decode_case(backend, canvas, case, config):
    digits = case.get("representation", "integer_token") == "decimal_digits"
    gold = expand_values(case["gold"]) if digits else case["gold"]
    forced = expand_values(case["forced_values"]) if digits else case["forced_values"]
    raw = decode(backend, canvas, gold, config.get("policy", "confidence"), forced,
                 config.get("starting_task", "A"))
    if not digits:
        return raw
    slots = case["slots"]

    def combine(values):
        result = {}
        for slot in slots:
            tens, ones = (values[key] for key in digit_keys(slot))
            result[slot] = None if tens is None or ones is None else 10 * tens + ones
        return result

    steps = []
    for step in raw["steps"]:
        current = combine(step["current_values"])
        candidates = {}
        # A semantic probe exists only while BOTH digits are masked. In
        # particular all unforced answers have probes on the first forward.
        for slot in slots:
            keys = digit_keys(slot)
            if all(key in step["candidates"] for key in keys):
                parts = [step["candidates"][key] for key in keys]
                scores = {str(v): mean((parts[0]["value_probabilities"][str(v // 10)],
                                       parts[1]["value_probabilities"][str(v % 10)])) for v in range(100)}
                candidates[slot] = {"token_id": [p["token_id"] for p in parts],
                                    "value": current[slot], "value_scores": scores,
                                    "gold_score": scores[str(case["gold"][slot])]}
        selected = step["selected_slot"].rsplit("_", 1)[0]
        steps.append({"step": step["step"], "current_values": current,
                      "current_correct": {s: current[s] == case["gold"][s] for s in slots},
                      "candidates": candidates, "selected_slot": selected,
                      "selected_digit_slot": step["selected_slot"],
                      "masks_before": step["masks_before"], "masks_after": step["masks_after"]})
    final = combine(raw["final_values"])
    return {"nfe": raw["nfe"], "initial_token_ids": raw["initial_token_ids"],
            "final_token_ids": raw["final_token_ids"], "final_values": final,
            "final_correct": {s: final[s] == case["gold"][s] for s in slots},
            "forced_slots": list(case["forced_values"]), "steps": steps,
            "score_kind": "mean_digit_probability", "token_trace": raw}
