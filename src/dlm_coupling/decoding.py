"""Controlled, one-token-at-a-time decoding on an immutable output canvas.

Model-specific logits alignment and tensor operations belong in the backend.
This module only schedules writes and records predictions *before* each write.
Ground-truth values are used to annotate records, never to choose tokens.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _probability(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite probability") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def decode(
    backend: Any,
    prepared: Any,
    gold: Mapping[str, int],
    policy: str = "confidence",
    forced_values: Mapping[str, int] | None = None,
    starting_task: str = "A",
) -> dict[str, Any]:
    """Fill every answer slot exactly once and return a JSON-compatible trace.

    ``prepared`` supplies ``token_ids``, ``slot_positions``, and per-slot
    ``value_token_ids`` (numeric-string to token-ID mappings). Every initial
    answer slot must contain ``backend.mask_id``; all other positions are fixed.

    ``backend.predict(tokens, positions, value_token_ids)`` returns one candidate
    dictionary per requested position, in that order. Its candidate confidence
    and requested token probabilities must come from the full vocabulary
    softmax; the candidate itself must exclude the mask token. No answer-domain
    restriction or model-specific logits shift is applied by this scheduler.

    Policies are global ``confidence`` or ``alternating`` (A/B tasks, then
    highest confidence within the chosen task). Absolute token position breaks
    confidence ties. A task with no remaining masks is skipped. Forced values
    are interventions inserted before the first forward; they cannot be changed.
    """
    if policy not in {"confidence", "alternating"}:
        raise ValueError("policy must be 'confidence' or 'alternating'")
    if starting_task not in {"A", "B"}:
        raise ValueError("starting_task must be 'A' or 'B'")
    mask_id = backend.mask_id
    if not _is_int(mask_id):
        raise ValueError("backend.mask_id must be an integer")
    token_ids = list(prepared.token_ids)
    positions = dict(prepared.slot_positions)
    value_ids = {slot: dict(values) for slot, values in prepared.value_token_ids.items()}
    if any(not _is_int(token) for token in token_ids):
        raise ValueError("token_ids must contain integers")
    if set(positions) != set(value_ids) or set(gold) != set(positions):
        raise ValueError("slot_positions, value_token_ids, and gold must have identical keys")
    if any(not isinstance(slot, str) or not slot for slot in positions):
        raise ValueError("slot keys must be nonempty strings")
    if any(not _is_int(pos) or not 0 <= pos < len(token_ids) for pos in positions.values()):
        raise ValueError("slot positions must be valid integer token indices")
    if len(set(positions.values())) != len(positions):
        raise ValueError("answer slots must have distinct positions")
    initial_masks = {index for index, token in enumerate(token_ids) if token == mask_id}
    if initial_masks != set(positions.values()):
        raise ValueError("the initial mask positions must be exactly the answer slots")

    slots = sorted(positions, key=positions.__getitem__)
    reverse_ids: dict[str, dict[int, int]] = {}
    for slot in slots:
        reverse: dict[int, int] = {}
        for value, token in value_ids[slot].items():
            try:
                number = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{slot}: numeric keys must be canonical strings in 0..99") from exc
            if not isinstance(value, str) or str(number) != value or not 0 <= number <= 99:
                raise ValueError(f"{slot}: numeric keys must be canonical strings in 0..99")
            if not _is_int(token) or token == mask_id:
                raise ValueError(f"{slot}: value token IDs must be integers other than mask_id")
            if token in reverse:
                raise ValueError(f"{slot}: different numeric values must have distinct token IDs")
            reverse[token] = number
        if not _is_int(gold[slot]) or str(gold[slot]) not in value_ids[slot]:
            raise ValueError(f"{slot}: gold must be an integer present in its numeric token map")
        reverse_ids[slot] = reverse

    task_of = {slot: slot.split(".", 1)[0] for slot in slots}
    if policy == "alternating" and any(task not in {"A", "B"} for task in task_of.values()):
        raise ValueError("alternating policy requires slot keys A.<name> or B.<name>")
    forced = dict(forced_values or {})
    if not set(forced).issubset(positions):
        raise ValueError("forced_values contains an unknown answer slot")
    for slot, value in forced.items():
        if not _is_int(value) or str(value) not in value_ids[slot]:
            raise ValueError(f"{slot}: forced value must be an integer present in its numeric token map")
        token_ids[positions[slot]] = value_ids[slot][str(value)]

    initial_token_ids = list(token_ids)
    forced_slots = [slot for slot in slots if slot in forced]
    remaining_count = len(slots) - len(forced_slots)
    steps: list[dict[str, Any]] = []
    next_task = starting_task

    for step_index in range(remaining_count):
        masked_slots = [slot for slot in slots if token_ids[positions[slot]] == mask_id]
        committed_before = [slot for slot in slots if slot not in masked_slots]
        predictions = backend.predict(
            list(token_ids),
            [positions[slot] for slot in masked_slots],
            [list(value_ids[slot].values()) for slot in masked_slots],
        )
        if len(predictions) != len(masked_slots):
            raise ValueError("backend returned a different number of candidates than requested positions")
        candidates: dict[str, dict[str, Any]] = {}
        for slot, prediction in zip(masked_slots, predictions):
            token = prediction["token_id"]
            if not _is_int(token) or token == mask_id:
                raise ValueError("backend candidates must be integer token IDs other than mask_id")
            probabilities = prediction["probabilities"]
            numeric_probabilities = {}
            for value, value_token in value_ids[slot].items():
                key = str(value_token)
                if key not in probabilities:
                    raise ValueError(f"{slot}: backend omitted a requested token probability ({key})")
                numeric_probabilities[value] = _probability(probabilities[key], f"{slot} probability")
            candidates[slot] = {
                "token_id": token,
                "token_text": prediction["token_text"],
                "value": reverse_ids[slot].get(token),
                "confidence": _probability(prediction["confidence"], f"{slot} confidence"),
                "gold_probability": numeric_probabilities[str(gold[slot])],
                "value_probabilities": numeric_probabilities,
                "raw_top_id": prediction["raw_top_id"],
                "raw_top_text": prediction["raw_top_text"],
            }

        current_values = {
            slot: candidates[slot]["value"] if slot in candidates else reverse_ids[slot].get(token_ids[positions[slot]])
            for slot in slots
        }
        eligible = masked_slots
        if policy == "alternating":
            eligible = [slot for slot in masked_slots if task_of[slot] == next_task]
            if not eligible:
                next_task = "B" if next_task == "A" else "A"
                eligible = [slot for slot in masked_slots if task_of[slot] == next_task]
        selected_slot = min(eligible, key=lambda slot: (-candidates[slot]["confidence"], positions[slot]))
        selected_token = candidates[selected_slot]["token_id"]
        masks_before = token_ids.count(mask_id)
        token_ids[positions[selected_slot]] = selected_token
        masks_after = token_ids.count(mask_id)
        if masks_after != masks_before - 1:
            raise AssertionError("each forward must remove exactly one mask")
        steps.append({
            "step": step_index,
            "masked_slots": masked_slots,
            "candidates": candidates,
            "current_values": current_values,
            "current_correct": {slot: current_values[slot] == gold[slot] for slot in slots},
            "committed_before": committed_before,
            "selected_slot": selected_slot,
            "selected_token_id": selected_token,
            "masks_before": masks_before,
            "masks_after": masks_after,
        })
        if policy == "alternating":
            next_task = "B" if task_of[selected_slot] == "A" else "A"

    final_values = {slot: reverse_ids[slot].get(token_ids[positions[slot]]) for slot in slots}
    if token_ids.count(mask_id) != 0 or len(steps) != remaining_count:
        raise AssertionError("decoding must exhaust the initially unforced masks")
    if any(token_ids[index] != token for index, token in enumerate(initial_token_ids) if index not in positions.values()):
        raise AssertionError("decoding changed the fixed canvas")
    if any(token_ids[positions[slot]] != initial_token_ids[positions[slot]] for slot in forced_slots):
        raise AssertionError("decoding changed an intervention value")
    return {
        "nfe": len(steps),
        "initial_token_ids": initial_token_ids,
        "final_token_ids": list(token_ids),
        "final_values": final_values,
        "final_correct": {slot: final_values[slot] == gold[slot] for slot in slots},
        "forced_slots": forced_slots,
        "steps": steps,
    }
