"""Paired arithmetic questions with controlled upstream dependencies.

A family contains six variants: two dependency conditions crossed with an
unchanged input and a +2 intervention on either source.  All conditions retain
both sources' facts, including the unused blue source in the shared condition.
The optional intermediate output reports a specified subtotal.  It does not
establish that the model internally used that subtotal to obtain its answer.

``depth`` counts additions on a path to a final answer, including the final
question-specific offset.  Fact generation and the executable graph use only
integers and addition; there is no Python evaluation of generated text.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

QUESTIONS = ("A", "B")
SLOTS = ("intermediate", "answer")
PERTURBATION_DELTA = 2


def _semantic_id(red: list[int], blue: list[int], offsets: dict[str, int]) -> str:
    # The first addition is commutative; swapping its two inputs does not
    # produce a new semantic program for the duplicate check.
    signature = {
        "red": sorted(red[:2]) + red[2:],
        "blue": sorted(blue[:2]) + blue[2:],
        "offsets": offsets,
    }
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _node_constant(node_id: str, value: int) -> dict[str, Any]:
    return {"id": node_id, "op": "const", "inputs": [], "value": value}


def _node_add(node_id: str, left: str, right: str, value: int) -> dict[str, Any]:
    return {"id": node_id, "op": "add", "inputs": [left, right], "value": value}


def _build_graph(
    receipts: dict[str, list[int]], offsets: dict[str, int], condition: str
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    subtotals: dict[str, str] = {}
    for source in ("red", "blue"):
        amounts = receipts[source]
        for index, value in enumerate(amounts):
            nodes.append(_node_constant(f"{source}.receipt{index}", value))
        subtotal = f"{source}.subtotal1"
        running = amounts[0] + amounts[1]
        nodes.append(_node_add(subtotal, f"{source}.receipt0", f"{source}.receipt1", running))
        for index, value in enumerate(amounts[2:], start=2):
            previous = subtotal
            subtotal = f"{source}.subtotal{index}"
            running += value
            nodes.append(_node_add(subtotal, previous, f"{source}.receipt{index}", running))
        subtotals[source] = subtotal

    outputs: dict[str, str] = {}
    for question in QUESTIONS:
        source = "red" if question == "A" or condition == "shared" else "blue"
        offset_id = f"{question}.offset"
        answer_id = f"{question}.answer"
        nodes.append(_node_constant(offset_id, offsets[question]))
        nodes.append(_node_add(answer_id, subtotals[source], offset_id, sum(receipts[source]) + offsets[question]))
        outputs[f"{question}.intermediate"] = subtotals[source]
        outputs[f"{question}.answer"] = answer_id

    edges = [[input_id, node["id"]] for node in nodes for input_id in node["inputs"]]
    return {"nodes": nodes, "edges": edges, "outputs": outputs}


def _fact_block(source: str, amounts: list[int]) -> list[str]:
    label = source.capitalize()
    return [
        f"{label} station received {amount} boxes in delivery {index + 1}."
        for index, amount in enumerate(amounts)
    ]


def _make_variant(
    family_id: str,
    condition: str,
    perturbation: str,
    red: list[int],
    blue: list[int],
    offsets: dict[str, int],
    semantic_id: str,
    split: str,
    seed: int,
    max_value: int = 99,
) -> dict[str, Any]:
    receipts = {"red": red.copy(), "blue": blue.copy()}
    if perturbation != "base":
        receipts[perturbation][0] += PERTURBATION_DELTA
    fact_blocks = {source: _fact_block(source, amounts) for source, amounts in receipts.items()}
    sources = {"A": "red", "B": "red" if condition == "shared" else "blue"}
    questions = {
        question: (
            f"{question}: Person {question} has {offsets[question]} more boxes than the total number "
            f"received by {sources[question]} station. How many boxes does Person {question} have?"
        )
        for question in QUESTIONS
    }
    graph = _build_graph(receipts, offsets, condition)
    values = {node["id"]: node["value"] for node in graph["nodes"]}
    gold = {slot: values[node_id] for slot, node_id in graph["outputs"].items()}
    affected = [] if perturbation == "base" else [q for q in QUESTIONS if sources[q] == perturbation]
    return {
        "variant_id": f"{family_id}.{condition}.{perturbation}",
        "condition": condition,
        "perturbation": perturbation,
        "facts": fact_blocks["red"] + fact_blocks["blue"],
        "questions": questions,
        "gold": gold,
        "affected_questions": affected,
        "graph": graph,
        "metadata": {
            "family_id": family_id,
            "split": split,
            "seed": seed,
            "semantic_id": semantic_id,
            "fact_blocks": fact_blocks,
            "receipts": receipts,
            "offsets": offsets.copy(),
            "question_sources": sources,
            "perturbed_node": None if perturbation == "base" else f"{perturbation}.receipt0",
            "perturbation_delta": 0 if perturbation == "base" else PERTURBATION_DELTA,
            "depth": len(red),
            "output_range": [0, max_value],
        },
    }


def _eligible(red: list[int], blue: list[int], offsets: dict[str, int]) -> bool:
    # In the independent condition, avoid accidental equality between an
    # intermediate/answer slot and another slot in any factual variant.
    # Shared intermediate slots intentionally have identical correct values.
    for source in ("base", "red", "blue"):
        red_total = sum(red) + (PERTURBATION_DELTA if source == "red" else 0)
        blue_total = sum(blue) + (PERTURBATION_DELTA if source == "blue" else 0)
        values = [red_total, red_total + offsets["A"], blue_total, blue_total + offsets["B"]]
        if len(set(values)) != len(values) or min(values) < 0 or max(values) > 99:
            return False
        if red_total + offsets["B"] > 99:
            return False
    return offsets["A"] != offsets["B"]


def generate_families(count: int, seed: int, split: str = "dev", depth: int = 2, max_value: int = 99) -> list[dict]:
    """Generate deterministic, unique families without consulting a model.

    The supported depth range is 2--12.  Deeper settings require revisiting
    the pilot's 0--99 answer domain instead of silently changing its task.
    ``seed`` controls a local RNG only; the split is included in that RNG's
    initialization.  Distinct calls may overlap, so persist ``semantic_id``
    and check it across split files when constructing a larger dataset.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(split, str) or not split or not all(c.isalnum() or c in "_-" for c in split):
        raise ValueError("split must be a nonempty identifier containing letters, digits, '_' or '-'")
    if isinstance(depth, bool) or not isinstance(depth, int) or not 2 <= depth <= 12:
        raise ValueError("depth must be an integer from 2 through 12")
    if type(max_value) is not int or not 9 <= max_value <= 99:
        raise ValueError("max_value must be an integer from 9 through 99")
    if count == 0:
        return []

    rng = random.Random(f"arithmetic-v1:{seed}:{split}:{depth}")
    families: list[dict] = []
    seen: set[str] = set()
    max_receipt = min(12, 70 // depth, max_value // depth)
    attempts = 0
    max_attempts = max(1000, count * 1000)
    while len(families) < count:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError("requested too many distinct families for this task domain; reduce count")
        red = [rng.randint(1, max_receipt) for _ in range(depth)]
        blue = [rng.randint(1, max_receipt) for _ in range(depth)]
        offsets = dict(zip(QUESTIONS, rng.sample(range(1, 7), k=2)))
        if not _eligible(red, blue, offsets):
            continue
        if max(sum(red) + max(offsets.values()) + 2, sum(blue) + offsets["B"] + 2) > max_value:
            continue
        semantic_id = _semantic_id(red, blue, offsets)
        if semantic_id in seen:
            continue
        seen.add(semantic_id)
        domain_tag = f".m{max_value}" if max_value != 99 else ""
        family_id = f"arithmetic.{split}.s{seed}.d{depth}{domain_tag}.{len(families):05d}"
        variants = [
            _make_variant(family_id, condition, perturbation, red, blue, offsets, semantic_id, split, seed, max_value)
            for condition in ("independent", "shared")
            for perturbation in ("base", "red", "blue")
        ]
        families.append({
            "family_id": family_id,
            "split": split,
            "seed": seed,
            "depth": depth,
            "max_value": max_value,
            "variants": variants,
        })
    return families


def validate_family(family: dict) -> None:
    """Check graph arithmetic, paired edits, truth labels, and split metadata.

    Raise ``ValueError`` for a malformed family. This also checks that saved
    factual text describes the graph's constants, avoiding a valid graph with
    an accidentally stale prompt. It does not compare this family against
    other families or certify any tokenizer-dependent matching constraint.
    """
    try:
        _validate_family(family)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError(f"malformed arithmetic family: {exc}") from exc


def _validate_family(family: dict) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    depth, seed, split = family["depth"], family["seed"], family["split"]
    # Reuse argument validation without sampling any examples.
    max_value = family.get("max_value", 99)
    generate_families(0, seed, split, depth, max_value)
    family_id = family["family_id"]
    domain_tag = f".m{max_value}" if max_value != 99 else ""
    prefix = f"arithmetic.{split}.s{seed}.d{depth}{domain_tag}."
    require(isinstance(family_id, str) and family_id.startswith(prefix), "family_id does not match split/seed/depth")
    require(family_id[len(prefix):].isdigit(), "family_id must end in a numeric family index")
    variants = family["variants"]
    require(isinstance(variants, list) and len(variants) == 6, "a family must contain six variants")
    lookup = {(v["condition"], v["perturbation"]): v for v in variants}
    expected_keys = {(c, p) for c in ("independent", "shared") for p in ("base", "red", "blue")}
    require(set(lookup) == expected_keys, "missing or duplicate condition/perturbation variants")
    base = lookup[("independent", "base")]
    red = base["metadata"]["receipts"]["red"]
    blue = base["metadata"]["receipts"]["blue"]
    offsets = base["metadata"]["offsets"]
    require(set(offsets) == set(QUESTIONS), "offsets must have exactly A and B")
    require(all(type(value) is int and 1 <= value <= 6 for value in offsets.values()), "invalid question offsets")
    for amounts in (red, blue):
        require(isinstance(amounts, list) and len(amounts) == depth, "receipt count must match depth")
        require(all(type(value) is int and value > 0 for value in amounts), "receipts must be positive integers")
    require(_eligible(red, blue, offsets), "family violates the bounded domain or distinct-slot constraint")
    semantic_id = _semantic_id(red, blue, offsets)
    expected_slots = {f"{q}.{slot}" for q in QUESTIONS for slot in SLOTS}

    for (condition, perturbation), variant in lookup.items():
        graph = variant["graph"]
        values: dict[str, int] = {}
        depths: dict[str, int] = {}
        edges = []
        for node in graph["nodes"]:
            node_id, inputs = node["id"], node["inputs"]
            require(node_id not in values, "graph node IDs must be unique")
            require(type(node["value"]) is int, "graph values must be integers")
            if node["op"] == "const":
                require(inputs == [], "constant nodes cannot have inputs")
                value, node_depth = node["value"], 0
            elif node["op"] == "add":
                require(len(inputs) == 2 and all(i in values for i in inputs), "add nodes need two preceding inputs")
                value = sum(values[i] for i in inputs)
                node_depth = max(depths[i] for i in inputs) + 1
            else:
                raise ValueError(f"unsupported graph operation: {node['op']}")
            require(node["value"] == value, "cached graph value disagrees with arithmetic")
            values[node_id], depths[node_id] = value, node_depth
            edges.extend([[i, node_id] for i in inputs])
        require(graph["edges"] == edges, "graph edges disagree with node inputs")
        require(set(graph["outputs"]) == expected_slots, "wrong graph output slots")
        truth = {slot: values[node_id] for slot, node_id in graph["outputs"].items()}
        require(truth == variant["gold"], "gold disagrees with graph evaluation")
        require(all(type(value) is int and 0 <= value <= max_value for value in variant["gold"].values()), "gold must contain bounded integers")
        for question in QUESTIONS:
            require(depths[graph["outputs"][f"{question}.answer"]] == depth, "incorrect final answer depth")
            require(depths[graph["outputs"][f"{question}.intermediate"]] == depth - 1, "incorrect intermediate depth")
        reference = _make_variant(family_id, condition, perturbation, red, blue, offsets, semantic_id, split, seed, max_value)
        for field in ("variant_id", "facts", "questions", "gold", "affected_questions", "graph"):
            require(variant[field] == reference[field], f"inconsistent variant field: {field}")
        for field, expected_value in reference["metadata"].items():
            require(variant["metadata"][field] == expected_value, f"inconsistent metadata: {field}")


def render_prompt(
    variant: dict,
    input_order: str = "AB",
    output_order: str = "AB",
    format_name: str = "answer_only",
) -> str:
    """Render a question without exposing any evaluator-only gold values.

    Input order jointly changes the source fact-block order (A=red, B=blue)
    and the question order. Output order independently changes the requested
    answer order. Thus this layout factor does not separate fact-position
    effects from question-position effects. Both fact blocks remain visible
    in the shared condition, where both questions refer to red station.
    """
    if input_order not in ("AB", "BA") or output_order not in ("AB", "BA"):
        raise ValueError("input_order and output_order must be 'AB' or 'BA'")
    if format_name not in ("answer_only", "with_intermediate"):
        raise ValueError("format_name must be 'answer_only' or 'with_intermediate'")
    sources = {"A": "red", "B": "blue"}
    lines = [
        "Use only the facts below. All counts are whole numbers of boxes.",
        "Facts:",
    ]
    for question in input_order:
        lines.extend(variant["metadata"]["fact_blocks"][sources[question]])
    lines.append("Questions:")
    lines.extend(variant["questions"][question] for question in input_order)
    if format_name == "with_intermediate":
        lines.append(
            "For each question, intermediate is the total number received by the station "
            "named in that question; answer is the person's final number of boxes."
        )
    lines.append("Return only the following format, replacing each <integer> with a number:")
    for question in output_order:
        if format_name == "answer_only":
            lines.append(f"{question}: answer=<integer>")
        else:
            lines.append(f"{question}: intermediate=<integer>; answer=<integer>")
    return "\n".join(lines)
