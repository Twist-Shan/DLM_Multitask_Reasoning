"""Small answer DAGs with exact, independently intervenable local definitions.

This is a new exploration distribution, not a replacement for precommit v2.
Graph reachability and value-changing counterfactuals are deliberately separate.
"""
from copy import deepcopy
from functools import reduce
import hashlib
import itertools
import json
import operator
import random

PROTOCOL = 'answer-graphs-v1'
FIELDS = 'ABCDE'
SYMBOLS = 'WXYZ'
FAMILIES = ('addition', 'subtraction', 'boolean', 'lookup')
GRAPHS = {
    'independent': ((), (), (), (), ()),
    'chain': ((), (0,), (1,), (2,), ()),
    'fork': ((), (0,), (0,), (0,), ()),
    'merge': ((), (), (0, 1), (2,), ()),
    'diamond': ((), (0,), (0,), (1, 2), ()),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stream(seed, *parts):
    return random.Random(int(digest([PROTOCOL, seed, *parts])[:16], 16))


def domain(family):
    if family in ('addition', 'subtraction'):
        return tuple(str(i) for i in range(10, 100))
    if family == 'boolean':
        return ('0', '1')
    if family == 'lookup':
        return tuple(SYMBOLS)
    raise ValueError(family)


def evaluate(case, override=None):
    """Evaluate equations in topological order, with optional do(node=value)."""
    override = override or {}
    if set(override) - set(FIELDS):
        raise ValueError('Unknown intervention field')
    values = {}
    for node in case['nodes']:
        name = node['name']
        if name in override:
            value = str(override[name])
        else:
            parents = [values[p] for p in node['parents']]
            params = node['params']
            if case['family'] == 'addition':
                value = str(sum(map(int, parents)) + sum(params))
            elif case['family'] == 'subtraction':
                value = str(sum(params) - sum(map(int, parents)) if parents else params[0] - params[1])
            elif case['family'] == 'boolean':
                inputs = list(map(int, parents))
                base = reduce({'AND': operator.and_, 'OR': operator.or_, 'XOR': operator.xor}[node['op']], inputs) if inputs else 0
                value = str(base ^ params[0] ^ params[1])
            elif case['family'] == 'lookup':
                key = ''.join(parents) if parents else node['input']
                value = node['table'][key]
            else:
                raise ValueError(case['family'])
        values[name] = value
    return values


def descendants(case, site):
    found = {site}
    for node in case['nodes']:
        if set(node['parents']) & found:
            found.add(node['name'])
    return found - {site}


def source(case, site, kind='change', alternative=None):
    """Source changes only the selected node's local definition.

    Numeric alternative is +/-1; Boolean uses bit flip; lookup supplies another
    symbol. same_value modifies operands or an unused table row, preserving all
    answers. No gold output is inserted into the answer canvas.
    """
    if kind not in ('change', 'same_value', 'self'):
        raise ValueError(kind)
    out = deepcopy(case)
    node = next(n for n in out['nodes'] if n['name'] == site)
    if kind == 'self':
        return out
    family = case['family']
    if family in ('addition', 'subtraction'):
        if alternative not in (-1, 1):
            raise ValueError('Numeric Sources require +/-1')
        node['params'][0] += alternative
        if kind == 'same_value':
            node['params'][1] += alternative if family == 'subtraction' and not node['parents'] else -alternative
    elif family == 'boolean':
        node['params'][0] ^= 1
        if kind == 'same_value':
            node['params'][1] ^= 1
    else:
        old = evaluate(case)
        if alternative not in SYMBOLS or alternative == old[site]:
            raise ValueError('Lookup Source needs a distinct symbol')
        key = ''.join(old[p] for p in node['parents']) if node['parents'] else node['input']
        if kind == 'same_value':
            # Match the exact old->new lexical edit at an inactive table row.
            key = next(k for k, v in node['table'].items() if k != key and v == old[site])
        node['table'][key] = alternative
    return out


def alternatives(case, site):
    if case['family'] in ('addition', 'subtraction'):
        return (-1, 1)
    if case['family'] == 'boolean':
        return (None,)
    return tuple(v for v in SYMBOLS if v != evaluate(case)[site])


def source_pairs(case):
    target = evaluate(case)
    for site in FIELDS:
        reachable = descendants(case, site)
        for alternative in alternatives(case, site):
            changed = source(case, site, 'change', alternative)
            changed_values = evaluate(changed)
            for kind in ('change', 'same_value'):
                src = changed if kind == 'change' else source(case, site, kind, alternative)
                actual = evaluate(src)
                yield dict(site=site, kind=kind, alternative=alternative, Source=src,
                           Target_values=target, Source_values=actual,
                           structural_descendants=sorted(reachable),
                           changed_fields=[k for k in FIELDS if actual[k] != target[k]],
                           candidate_values=changed_values)


def validate(case):
    if case['family'] not in FAMILIES or case['graph'] not in GRAPHS:
        raise ValueError('Unknown family/graph')
    names = [n['name'] for n in case['nodes']]
    if sorted(names) != list(FIELDS) or sorted(case['definition_order']) != list(FIELDS):
        raise ValueError('Five unique node labels required')
    known = set()
    for i, n in enumerate(case['nodes']):
        if n['parents'] != [names[j] for j in GRAPHS[case['graph']][i]]:
            raise ValueError('Graph mismatch')
        if not set(n['parents']) <= known:
            raise ValueError('Not a DAG in topological order')
        known.add(n['name'])
    values = evaluate(case)
    if any(v not in domain(case['family']) for v in values.values()):
        raise ValueError('Answer width/domain changed')
    if case['family'] in ('addition', 'subtraction'):
        lo, hi = (1, 9) if case['family'] == 'addition' else (10, 99)
        if any(not lo <= p <= hi for n in case['nodes'] for p in n['params']):
            raise ValueError('Operand width changed')
    for pair in source_pairs(case):
        actual = pair['Source_values']
        if any(v not in domain(case['family']) for v in actual.values()):
            raise ValueError('Source answer domain changed')
        if case['family'] in ('addition', 'subtraction'):
            if any(not lo <= p <= hi for n in pair['Source']['nodes'] for p in n['params']):
                raise ValueError('Source operand width changed')
        if pair['kind'] == 'same_value' and actual != values:
            raise AssertionError('same_value changed an answer')
        site = pair['site']
        if pair['kind'] == 'change':
            if actual[site] == values[site] or evaluate(case, {site: actual[site]}) != actual:
                raise AssertionError('Source is not a local value intervention')
            if set(pair['changed_fields']) - ({site} | set(pair['structural_descendants'])):
                raise AssertionError('A non-descendant changed')
    return True


def make_case(family, graph, rng, labels, definition_order):
    case = dict(family=family, graph=graph, nodes=[], definition_order=list(definition_order))
    for i, parents in enumerate(GRAPHS[graph]):
        node = dict(name=labels[i], parents=[labels[p] for p in parents], params=[])
        if family == 'addition':
            node['params'] = [rng.randint(2, 6) for _ in range(2 if parents else 4)]
        elif family == 'subtraction':
            desired = rng.randint(14, 26)
            if parents:
                total = sum(int(evaluate(case)[p]) for p in node['parents']) + desired
                first = rng.randint(11, total - 11)
                node['params'] = [first, total - first]
            else:
                second = rng.randint(12, 25)
                node['params'] = [second + desired, second]
        elif family == 'boolean':
            node.update(params=[rng.randrange(2), rng.randrange(2)], op=rng.choice(('AND', 'OR', 'XOR')) if len(parents) > 1 else 'XOR')
        else:
            keys = [''.join(k) for k in itertools.product(SYMBOLS, repeat=max(1, len(parents)))]
            node.update(input=rng.choice(SYMBOLS), table={k: rng.choice(SYMBOLS) for k in keys})
            values = evaluate(case)
            active = ''.join(values[p] for p in node['parents']) if parents else node['input']
            duplicate = rng.choice([k for k in keys if k != active])
            node['table'][duplicate] = node['table'][active]
        case['nodes'].append(node)
    return case


def dataset(seed=123, per_cell=10, families=FAMILIES, graphs=tuple(GRAPHS)):
    if per_cell < 1 or per_cell > 1000 or set(families) - set(FAMILIES) or set(graphs) - set(GRAPHS):
        raise ValueError('Invalid dataset configuration')
    cases, used = [], set()
    for family in families:
        for graph in graphs:
            for index in range(per_cell):
                rng = stream(seed, 'exploration', family, graph, index)
                layout_rng = stream(seed, 'labels', graph, index)
                labels = layout_rng.sample(list(FIELDS), len(FIELDS))
                definitions = layout_rng.sample(list(FIELDS), len(FIELDS))
                for _ in range(10000):
                    case = make_case(family, graph, rng, labels, definitions)
                    try:
                        validate(case)
                    except ValueError:
                        continue
                    keys = {digest(prompt(case))} | {digest(prompt(p['Source'])) for p in source_pairs(case)}
                    if keys & used:
                        continue
                    used.update(keys)
                    case.update(id=f'ag1-{family}-{graph}-{index:03d}', split='exploration', instance_sha256=digest(case), input_sha256=digest(prompt(case)))
                    cases.append(case)
                    break
                else:
                    raise RuntimeError('Could not generate disjoint input families')
    return dict(protocol=PROTOCOL, seed=seed, per_cell=per_cell, families=list(families), graphs=list(graphs), cases=cases,
                scope='New exploratory questions; no prior confirmation items generated or evaluated; no model-based filtering')


def prompt(case, requested=FIELDS):
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(FIELDS):
        raise ValueError('Invalid requested fields')
    intro = {
        'addition': 'Use ordinary integer arithmetic.',
        'subtraction': 'Use ordinary integer arithmetic. Subtract the entire parenthesized sum.',
        'boolean': 'Values are bits 0 or 1. AND is 1 only if both inputs are 1. OR is 1 if at least one input is 1. XOR is 1 if the inputs differ. Evaluate parentheses first.',
        'lookup': 'W, X, Y, Z are symbols, not numbers. Use only the supplied lookup tables. For a two-input table, the ordered input pair selects one row.',
    }[case['family']]
    nodes = {n['name']: n for n in case['nodes']}
    lines = [intro, 'Definitions can refer to definitions printed later. There are no circular dependencies.']
    for name in case['definition_order']:
        n = nodes[name]
        pp, parents = n['params'], n['parents']
        if case['family'] == 'addition':
            local = f'({pp[0]} + {pp[1]})' if parents else f'({pp[0]} + {pp[1]}) + ({pp[2]} + {pp[3]})'
            expression = ' + '.join([*parents, local])
        elif case['family'] == 'subtraction':
            expression = f'({pp[0]} + {pp[1]}) - (' + ' + '.join(parents) + ')' if parents else f'{pp[0]} - {pp[1]}'
        elif case['family'] == 'boolean':
            local = f'({pp[0]} XOR {pp[1]})'
            joined = (' ' + n['op'] + ' ').join(parents)
            expression = f'({joined}) XOR {local}' if parents else local
        else:
            rows = '; '.join(f'({", ".join(key)}) -> {value}' for key, value in n['table'].items())
            lines.append(f'Table for {name}: {rows}.')
            expression = f'lookup({", ".join(parents) if parents else n["input"]}) in the table for {name}'
        lines.append(f'Definition of {name}: {name} = {expression}.')
    lines.extend([f'Requested outputs: {", ".join(requested)}.',
                  'Fill only the requested final values in the supplied answer template. No explanation. Do not output intermediate values.'])
    return '\n'.join(lines)


def layouts(case):
    topological = ''.join(n['name'] for n in case['nodes'])
    return dict(topological=topological, reverse=topological[::-1])


class CanvasBuilder:
    """Audit all domain values in the scaffold, then exact prefix concatenation.

    Reusing scaffold audits avoids retokenizing a long prompt for every value.
    A tokenizer that merges across the native assistant boundary is rejected.
    """
    def __init__(self, backend):
        self.b = backend
        self.templates = {}

    def build(self, case, requested, layout):
        from .precommit_v2 import native_suffix
        if sorted(requested) != sorted(layout) or len(set(layout)) != len(layout):
            raise ValueError('Requested outputs and slots must agree')
        tok = self.b.tokenizer
        user = prompt(case, requested)
        prefix = tok.apply_chat_template([dict(role='user', content=user)], tokenize=False, add_generation_prompt=True)
        suffix = native_suffix(tok, user, prefix)
        key = (case['family'], layout, suffix)
        if key not in self.templates:
            vals = domain(case['family'])
            scaffold = '; '.join(f'{name} = {vals[0]}' for name in layout) + '.'
            text = scaffold + suffix
            base = list(tok.encode(text, add_special_tokens=False))
            decoded = [tok.decode(base[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in range(len(base) + 1)]
            if decoded[-1] != text:
                raise ValueError('Scaffold round trip failed')
            spans, maps = {}, {}
            for name in layout:
                start = scaffold.index(f'{name} = ') + len(f'{name} = ')
                stop = start + len(vals[0])
                # Letter tokens can include their preceding space (e.g. " W").
                # Expand only into whitespace, never into the field label/delimiter.
                boundaries = [(i, len(s)) for i, s in enumerate(decoded) if text.startswith(s)]
                left = max((p for p in boundaries if p[1] <= start), key=lambda p: p[1])
                right = min((p for p in boundaries if p[1] >= stop), key=lambda p: p[1])
                if text[left[1]:start].strip() or text[stop:right[1]].strip():
                    raise ValueError('Value token crosses non-whitespace scaffold boundary')
                positions = list(range(left[0], right[0]))
                width = 2 if case['family'] in ('addition', 'subtraction') else 1
                if len(positions) != width:
                    raise ValueError(f'{name}: expected {width} value-token positions')
                mapping = {}
                for value in vals:
                    trial = list(tok.encode(text[:start] + value + text[stop:], add_special_tokens=False))
                    if len(trial) != len(base) or any(a != b for i, (a, b) in enumerate(zip(base, trial)) if i not in positions):
                        raise ValueError('Value changed scaffold tokenization outside its slot')
                    mapping[value] = [trial[i] for i in positions]
                spans[name], maps[name] = positions, mapping
            self.templates[key] = dict(base=base, scaffold=scaffold, spans=spans, maps=maps)
        template = self.templates[key]
        prefix_ids = list(tok.encode(prefix, add_special_tokens=False))
        ids = list(tok.encode(prefix + template['scaffold'] + suffix, add_special_tokens=False))
        if ids != prefix_ids + template['base'] or self.b.mask_id in ids:
            raise ValueError('Assistant boundary merge or prompt mask collision')
        offset = len(prefix_ids)
        spans = {k: [offset + p for p in ps] for k, ps in template['spans'].items()}
        for positions in spans.values():
            for p in positions:
                ids[p] = self.b.mask_id
        return dict(ids=ids, spans=spans, maps=template['maps'], prompt=user, layout=layout, requested=requested,
                    prompt_length=offset, native_suffix=suffix, output_scaffold=template['scaffold'], unused=[],
                    uses_gold_to_prepare=False)


def read_values(canvas, ids):
    return {name: next((v for v, tokens in canvas['maps'][name].items() if tokens == [ids[p] for p in positions]), None)
            for name, positions in canvas['spans'].items()}


def predict(backend, canvas, ids, positions):
    probes = {}
    for name, span in canvas['spans'].items():
        for j, p in enumerate(span):
            probes[p] = sorted({tokens[j] for tokens in canvas['maps'][name].values()})
    return backend.predict(ids, positions, [probes[p] for p in positions])


def decode(backend, canvas, policy, initial=None):
    if policy not in ('confidence', 'fixed_layout'):
        raise ValueError(policy)
    ids = canvas['ids'].copy()
    positions = [p for span in canvas['spans'].values() for p in span]
    writes = []
    while len(writes) < len(positions):
        remaining = [p for p in positions if ids[p] == backend.mask_id]
        predictions = initial if not writes and initial is not None else predict(backend, canvas, ids, remaining)
        j = max(range(len(remaining)), key=lambda i: (predictions[i]['confidence'], -remaining[i])) if policy == 'confidence' else 0
        p, chosen = remaining[j], predictions[j]
        if chosen['token_id'] == backend.mask_id:
            raise ValueError('Cannot commit MASK')
        writes.append(dict(position=p, remaining=remaining, predictions=predictions, token_id=chosen['token_id']))
        ids[p] = chosen['token_id']
    if any(a != b for i, (a, b) in enumerate(zip(ids, canvas['ids'])) if i not in positions):
        raise AssertionError('Modified immutable position')
    return dict(policy=policy, values=read_values(canvas, ids), ids=ids, writes=writes,
                logical_forwards=len(positions), new_forwards=len(positions) - int(initial is not None))
