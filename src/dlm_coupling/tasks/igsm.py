"""Multi-target extension of official iGSM graphs (arithmetic modulo 23)."""
import json
import re


def references(expr):
    if "ref" in expr:
        return {expr["ref"]}
    return set().union(*(references(x) for x in expr.get("args", [])))


def evaluate_graph(nodes, modulus=23):
    values, active = {}, set()
    def evaluate(key):
        if key in active:
            raise ValueError("Cyclic iGSM expression graph")
        if key not in values:
            active.add(key)
            value = expression(nodes[key]["expression"])
            values[key] = value if modulus is None else value % modulus
            active.remove(key)
        return values[key]
    def expression(e):
        if "ref" in e:
            return evaluate(e["ref"])
        if "literal" in e:
            return e["literal"]
        args = [expression(x) for x in e["args"]]
        if e["op"] == "mul":
            if len(args) != 2: raise ValueError("mul needs two arguments")
            return args[0] * args[1]
        if e["op"] == "diff":
            if len(args) != 2: raise ValueError("diff needs two arguments")
            return args[0] - args[1]
        if e["op"] == "sum": return sum(args)
        raise ValueError("Unknown expression operation")
    for key in nodes: evaluate(key)
    return values


def ancestors(nodes, key):
    found = set()
    todo = list(references(nodes[key]["expression"]))
    while todo:
        node = todo.pop()
        if node == key: raise ValueError("Cyclic dependency")
        if node not in found:
            found.add(node)
            todo.extend(references(nodes[node]["expression"]))
    return found


def relation(nodes, a, b):
    aa, ab = ancestors(nodes, a), ancestors(nodes, b)
    if a in ab: return "downstream"
    if b in aa: return "reverse_downstream"
    common = aa & ab
    if not common: return "independent"
    # Sharing a known constant alone does not count as shared computation.
    if any(references(nodes[n]["expression"]) and operation_count(nodes[n]["expression"]) for n in common):
        return "shared"
    return "shared_constants_only"


def operation_count(e):
    if "args" not in e: return 0
    return max(0, len(e["args"]) - 1) + sum(operation_count(x) for x in e["args"])


def validate_pair(pair):
    values = evaluate_graph(pair["nodes"])
    if any(values[k] != node["value"] for k, node in pair["nodes"].items()):
        raise ValueError("Independent evaluator disagrees with upstream lookup")
    a, b = (pair["targets"][q]["node"] for q in ("A", "B"))
    if a == b or relation(pair["nodes"], a, b) != pair["relation"]:
        raise ValueError("Incorrect pair relation")
    for q in ("A", "B"):
        if pair["targets"][q]["gold"] != values[pair["targets"][q]["node"]]:
            raise ValueError("Incorrect target gold")


def render(pair, order):
    if order not in ("A", "B", "AB", "BA"):
        raise ValueError("Invalid question order")
    lines = ["Solve the questions using the facts below. All arithmetic is modulo 23: "
             "after each calculation, keep the remainder from 0 to 22. "
             "For example, 20 + 5 = 2 and 2 - 5 = 20 in this arithmetic.",
             "Facts:", pair["facts"], "Questions:"]
    lines.extend(q + ": " + pair["targets"][q]["question"] for q in order)
    lines.append('Return only a JSON object with keys ' + ', '.join('"' + q + '"' for q in order)
                 + ' and integer answers from 0 to 22. Do not include explanations or other keys.')
    return "\n".join(lines)


def parse_answers(text, labels):
    """Strict JSON; allow a single code fence, reject duplicate keys/trailing prose."""
    text = text.strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if not match: return {}, "format"
        text = match.group(1)
    def unique(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError("duplicate key")
            result[key] = value
        return result
    try:
        result = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError): return {}, "format"
    if not isinstance(result, dict) or set(result) != set(labels): return {}, "keys"
    if any(type(v) is not int or not 0 <= v < 23 for v in result.values()): return {}, "domain"
    return result, None
