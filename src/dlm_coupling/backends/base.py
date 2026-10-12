"""Backend contract and tokenizer checks; imports no GPU dependencies."""

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Protocol


@dataclass
class PreparedCanvas:
    token_ids: list[int]
    slot_positions: dict[str, int]
    value_token_ids: dict[str, dict[str, int]]
    prompt_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    mask_id: int
    metadata: dict[str, Any]

    def prepare(self, prompt: str, slot_keys: list[str]) -> PreparedCanvas: ...

    def predict(
        self,
        token_ids: list[int],
        positions: list[int],
        value_token_ids: list[list[int]],
    ) -> list[dict[str, Any]]: ...

    def decode_token(self, token_id: int) -> str: ...


def output_template(slot_keys: list[str], values: list[str]) -> str:
    """Render task-major fields without changing task or field order."""
    if not slot_keys or len(slot_keys) != len(values):
        raise ValueError("Provide a nonempty, equally sized list of slots and values.")
    if len(set(slot_keys)) != len(slot_keys):
        raise ValueError("Slot keys must be unique.")
    tasks: dict[str, list[str]] = {}
    previous_task = None
    for key, value in zip(slot_keys, values):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid task.field slot key: {key!r}")
        task, field_name = key.split(".")
        if task in tasks and task != previous_task:
            raise ValueError("Slots must be grouped by task, matching output order.")
        tasks.setdefault(task, []).append(f"{field_name}={value}")
        previous_task = task
    return "\n".join(f"{task}: {'; '.join(fields)}" for task, fields in tasks.items()) + "\n"


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def build_canvas(
    tokenizer: Any, prompt: str, slot_keys: list[str], mask_id: int
) -> PreparedCanvas:
    """Check the full chat text for one-token, context-stable values 0..99.

    No answer is supplied here. The domain is fixed in advance, and does not
    constrain model predictions. Multi-token values are reported as unavailable.
    """
    mask_text = tokenizer.decode(
        [mask_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if not mask_text or _encode(tokenizer, mask_text) != [mask_id]:
        raise ValueError("The configured mask token does not round-trip.")
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    if not isinstance(chat_text, str):
        raise TypeError("apply_chat_template(tokenize=False) must return text.")
    prompt_ids = _encode(tokenizer, chat_text)
    if mask_id in prompt_ids:
        raise ValueError("The input prompt must not contain the model mask token.")
    template = output_template(slot_keys, [mask_text] * len(slot_keys))
    full_text = chat_text + template
    token_ids = _encode(tokenizer, full_text)
    if token_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat/scaffold boundary changes prompt tokenization; revise the template.")
    positions = [i for i, token_id in enumerate(token_ids) if token_id == mask_id]
    if len(positions) != len(slot_keys):
        raise ValueError("The scaffold must contain exactly one mask per answer slot.")
    round_trip = tokenizer.decode(
        token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if _encode(tokenizer, round_trip) != token_ids:
        raise ValueError("The full canvas does not preserve token IDs under round-trip.")
    value_maps: dict[str, dict[str, int]] = {}
    for index, (key, position) in enumerate(zip(slot_keys, positions)):
        value_maps[key] = {}
        for value in range(100):
            replacements = [mask_text] * len(slot_keys)
            replacements[index] = str(value)
            candidate = _encode(tokenizer, chat_text + output_template(slot_keys, replacements))
            if (
                len(candidate) == len(token_ids)
                and candidate[:position] == token_ids[:position]
                and candidate[position + 1 :] == token_ids[position + 1 :]
                and candidate[position] != mask_id
            ):
                value_maps[key][str(value)] = candidate[position]
        if not value_maps[key]:
            raise ValueError(f"No stable one-token integer in 0..99 at slot {key}.")
        if len(set(value_maps[key].values())) != len(value_maps[key]):
            raise ValueError(f"Different numeric values alias to the same token at slot {key}.")
    return PreparedCanvas(
        token_ids=token_ids,
        slot_positions=dict(zip(slot_keys, positions)),
        value_token_ids=value_maps,
        prompt_length=len(prompt_ids),
        metadata={
            "output_template": template,
            "canvas_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
            "candidate_domain": "integers 0..99; context-stable one-token subset",
            "decoding_constraint": "mask token excluded; all other vocabulary tokens allowed",
            "uses_gold_to_prepare": False,
            "output_token_count": len(token_ids) - len(prompt_ids),
            "fixed_scaffold": True,
        },
    )


def raw_logit_positions(positions: list[int], family: str) -> list[int]:
    """Dream's official sampler shifts raw logits right exactly once."""
    if family not in {"illada", "dream"}:
        raise ValueError(f"Unsupported model family: {family}")
    if any(position < 0 for position in positions):
        raise ValueError("Positions must be nonnegative.")
    return [max(0, position - 1) for position in positions] if family == "dream" else list(positions)


def validate_prediction_request(
    token_ids: list[int], positions: list[int], value_token_ids: list[list[int]]
) -> None:
    if len(positions) != len(value_token_ids):
        raise ValueError("One probe-token list is required per requested position.")
    if len(set(positions)) != len(positions):
        raise ValueError("Prediction positions must be unique.")
    if any(position < 0 or position >= len(token_ids) for position in positions):
        raise ValueError("Prediction position lies outside the canvas.")
