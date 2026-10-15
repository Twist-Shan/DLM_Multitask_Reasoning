"""Addition-only calibration v2. Keep the v1 protocol and data unchanged."""
import math
import random
import re

from . import precommit as v1

RELATIONS = v1.RELATIONS
SOURCE_KINDS = v1.SOURCE_KINDS


def gold(p, relation, c=None):
    if relation == 'copy':
        return dict(A=p['q'], B=p['q'])
    c = p['x'] + p['y'] if c is None else c
    d, u, v = p['w']+p['z'], p['u']+p['r'], p['v']+p['s']
    a = c + u
    return dict(A=a, B={'independent': d+v, 'shared': c+v, 'downstream': a+v}[relation])


def source(p, relation, kind):
    q = v1.source(p, relation, kind)
    if relation == 'shared' and kind == 'diff_C_same_answers':
        q['v'] = p['v'] - 1
    return q


def counterfactual(p, relation, kind):
    if relation == 'copy':
        return gold(source(p, relation, kind), relation)
    return gold(p, relation, c=p['x']+p['y']+1) if kind in ('C_only', 'diff_C_same_answers') else gold(p, relation)


def prompt(p, relation, requested='AB', style='numeric'):
    # Changing precisely these expressions leaves instructions and facts identical.
    return v1.prompt(p, relation, requested, style).replace('2 * ', '')


def variant_keys(p, source_fn=source):
    return {tuple(source_fn(p, rel, kind)[k] for k in v1.FACTS)
            for rel in RELATIONS for kind in SOURCE_KINDS}


def dataset(seed=123, sizes=None):
    sizes = sizes or dict(calibration=30, discovery=30, confirmation=100)
    # A new stream alone does not guarantee disjointness. Exclude all v1 variants.
    old = v1.dataset(seed)
    used = set().union(*(variant_keys(g['p'], v1.source) for g in old['groups']))
    historical = v1.old_six_fact_keys()
    groups, streams = [], {}
    for split, n in sizes.items():
        stream = int(v1.fingerprint(['precommit-v2', seed, split])[:16], 16)
        streams[split] = stream
        rng = random.Random(stream)
        made = 0
        for _ in range(200000):
            if made == n:
                break
            p = dict(x=rng.randint(3,6), y=rng.randint(5,8), w=rng.randint(3,8),
                     z=rng.randint(5,8), u=rng.randint(3,6), r=rng.randint(5,8),
                     v=rng.randint(4,6), s=rng.randint(5,8))
            variants = [source(p, rel, kind) for rel in RELATIONS for kind in SOURCE_KINDS]
            keys = variant_keys(p)
            if keys & used or {tuple(q[k] for k in v1.SIX_FACTS) for q in variants} & historical:
                continue
            if any(not 10 <= value <= 99 for rel in RELATIONS for q in variants
                   for value in gold(q, rel).values()):
                continue
            used.update(keys)
            groups.append(dict(id=f'v2-{split}-{made:03d}', split=split, p=p,
                               instance_sha256=v1.fingerprint(p)))
            made += 1
        if made != n:
            raise RuntimeError('Could not construct disjoint v2 groups')
    return dict(master_seed=seed, protocol='precommit-v2', streams=streams, groups=groups,
                v1_dataset_sha256=v1.fingerprint(old),
                exclusion='All v1 groups and Source variants, historical six-input variants; v2 splits disjoint including Sources')


def native_suffix(tokenizer, user, prefix):
    marker = '__DLM_ASSISTANT_BOUNDARY__'
    if marker in user:
        raise ValueError('Assistant boundary marker collision')
    closed = tokenizer.apply_chat_template(
        [dict(role='user', content=user), dict(role='assistant', content=marker)],
        tokenize=False, add_generation_prompt=False)
    if not closed.startswith(prefix + marker):
        raise ValueError('Native assistant prefix differs from generation prefix')
    suffix = closed[len(prefix + marker):]
    if not suffix:
        raise ValueError('Native chat template has no assistant ending')
    return suffix


class CanvasBuilder:
    def __init__(self, backend, task='additive'):
        if task not in ('additive', 'legacy'):
            raise ValueError(task)
        self.b, self.task, self.cache = backend, task, {}

    def build(self, p, relation, requested='AB', style='numeric', unused=False, ending='native'):
        if ending not in ('bare', 'native'):
            raise ValueError(ending)
        key = (tuple(sorted(p.items())), relation, requested, style, unused, ending)
        if key in self.cache:
            return self.cache[key]
        b, tok = self.b, self.b.tokenizer
        user = (prompt if self.task == 'additive' else v1.prompt)(p, relation, requested, style)
        prefix = tok.apply_chat_template([dict(role='user', content=user)], tokenize=False,
                                         add_generation_prompt=True)
        suffix = native_suffix(tok, user, prefix) if ending == 'native' else ''
        scaffold = '; '.join(f'{k} = 10' for k in requested) + '.'
        text = prefix + scaffold + suffix
        ids = list(tok.encode(text, add_special_tokens=False))
        decoded = [tok.decode(ids[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                   for i in range(len(ids)+1)]
        if decoded[-1] != text or b.mask_id in ids:
            raise ValueError('Canvas round trip / mask collision')
        boundary_text = prefix + scaffold
        if boundary_text not in decoded:
            raise ValueError('Scaffold/suffix boundary is not token aligned')
        insertion = decoded.index(boundary_text)
        spans, mappings = {}, {}
        for m in re.finditer(r'([AB]) = (10)', scaffold):
            field = m[1]
            lo, hi = len(prefix)+m.start(2), len(prefix)+m.end(2)
            if text[:lo] not in decoded or text[:hi] not in decoded:
                raise ValueError(f'{field}: numeral boundaries are not token aligned')
            positions = list(range(decoded.index(text[:lo]), decoded.index(text[:hi])))
            mapping = {}
            for value in range(10,100):
                trial = list(tok.encode(text[:lo]+str(value)+text[hi:], add_special_tokens=False))
                if len(trial) != len(ids) or any(a != c for i, (a,c) in enumerate(zip(ids,trial)) if i not in positions):
                    raise ValueError(f'{field}: domain-wide tokenization fails at {value}')
                mapping[value] = [trial[i] for i in positions]
            spans[field], mappings[field] = positions, mapping
        if set(spans) != set(requested) or len({len(x) for x in spans.values()}) != 1:
            raise ValueError('Unequal or missing answer widths')
        suffix_ids = ids[insertion:].copy()
        for positions in spans.values():
            for i in positions:
                ids[i] = b.mask_id
        neutral = []
        if unused:
            width = len(next(iter(spans.values())))
            neutral = list(range(insertion, insertion+width))
            ids[insertion:insertion] = [b.mask_id]*width
        suffix_positions = list(range(insertion+len(neutral), len(ids)))
        assert [ids[i] for i in suffix_positions] == suffix_ids
        result = dict(ids=ids, spans=spans, maps=mappings, unused=neutral, prompt=user,
            requested=requested, output_scaffold=scaffold, ending=ending,
            native_suffix=suffix, suffix_ids=suffix_ids, suffix_positions=suffix_positions,
            neutral_policy='unrequested masks after scaffold and before native suffix; held masked',
            task=self.task, uses_gold_to_prepare=False)
        self.cache[key] = result
        return result


def wilson_lower(k, n, z=1.959963984540054):
    if not n or not 0 <= k <= n:
        raise ValueError('Wilson input must satisfy 0 <= k <= n and n > 0')
    p = k/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def cluster_interval(rows, seed=123, repetitions=4000):
    pairs = {}
    for row in rows:
        pairs.setdefault(row['case'], []).append(float(row['correct']))
    means = [sum(v)/len(v) for _,v in sorted(pairs.items())]
    if not means:
        raise ValueError('Empty cluster sample')
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(means, k=len(means)))/len(means) for _ in range(repetitions))
    return [draws[int(.025*(repetitions-1))], draws[int(.975*(repetitions-1))]]


def score_gate(records, expected_groups=30, lower=.7, accuracy=.8, validity=.95):
    cells = {}
    for rel in RELATIONS:
        for req in ('A','B'):
            rows = [r for r in records if r['relation']==rel and r['requested']==req]
            keys = {(r['case'],r['unused']) for r in rows}
            cases = {r['case'] for r in rows}
            if len(cases)!=expected_groups or len(rows)!=2*expected_groups or len(keys)!=len(rows):
                raise ValueError('Single-question gate needs one complete paired canvas per group')
            if any((case, unused) not in keys for case in cases for unused in (False, True)):
                raise ValueError('Incomplete paired canvas')
            n, k = len(rows), sum(r['correct'] for r in rows)
            valid = sum(r['valid'] for r in rows)/n
            lb = wilson_lower(k,n)
            cells[f'{rel}/{req}'] = dict(outputs=n,groups=len(cases),correct=k,accuracy=k/n,
                validity=valid,pooled_wilson_lower=lb,
                cluster_bootstrap_95=cluster_interval(rows),
                pass_gate=(k/n>=accuracy and lb>=lower and valid>=validity))
    return dict(pass_gate=all(c['pass_gate'] for c in cells.values()),cells=cells,
        accuracy_threshold=accuracy,pooled_wilson_threshold=lower,validity_threshold=validity,
        selected_style='numeric',
        interpretation='Operational gate only: pooled Wilson assumes independent outputs, violated by paired canvases. Do not claim nominal 95% coverage. Cluster bootstrap resamples underlying groups. AB/BA and individual unused conditions never gate.')
