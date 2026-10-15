"""Pre-commitment arithmetic protocol; no implicit hidden-state carry between forwards.

All model outputs use the full vocabulary. Values used for scoring never enter a
masked canvas. The primary causal measurement is a single frozen token state.
"""
from contextlib import ExitStack
import hashlib
import json
import math
import random
import re

from .backends.base import raw_logit_positions
from .shared_causal import ResidualPatch, read_values

RELATIONS = ('independent', 'shared', 'downstream')
SOURCE_KINDS = ('self', 'same_C_same_answers', 'same_C_diff_answers',
                'diff_C_same_answers', 'C_only')
FACTS = ('x', 'y', 'w', 'z', 'u', 'r', 'v', 's')
SIX_FACTS = ('x', 'y', 'u', 'r', 'v', 's')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def gold(p, relation, c=None):
    if relation == 'copy':
        return dict(A=p['q'], B=p['q'])
    if relation not in RELATIONS:
        raise ValueError(relation)
    c = p['x'] + p['y'] if c is None else c
    d, u, v = p['w'] + p['z'], p['u'] + p['r'], p['v'] + p['s']
    a = c + u
    return dict(A=a, B={'independent': 2*d+v, 'shared': 2*c+v,
                        'downstream': 2*a+v}[relation])


def source(p, relation, kind):
    q = dict(p)
    if relation == 'copy':
        if kind == 'C_only': q['q'] += 1
        elif kind != 'self': raise ValueError('Copy control only supports self / C_only')
    elif kind == 'self': pass
    elif kind == 'same_C_same_answers': q.update(x=p['x']-1, y=p['y']+1)
    elif kind == 'same_C_diff_answers': q.update(u=p['u']+2, v=p['v']+3)
    elif kind == 'diff_C_same_answers':
        q.update(x=p['x']+1, u=p['u']-1, v=p['v']-(2 if relation=='shared' else 0))
    elif kind == 'C_only': q['x'] += 1
    else: raise ValueError(kind)
    return q


def counterfactual(p, relation, kind):
    if relation == 'copy': return gold(source(p, relation, kind), relation)
    return gold(p, relation, c=p['x']+p['y']+1) if kind in ('C_only','diff_C_same_answers') else gold(p, relation)


def old_six_fact_keys():
    # Include every six-fact variant from prior local generators, not only bases.
    from .implicit_sharing import data, sources
    from .mechanism_pilot import dataset
    from .headbank_pilot import validation_cases
    from .baseline_repair import splits
    from .shared_followup import cases
    values = [p for c in dataset()+validation_cases() for p in c['variants'].values()]
    values += [c['base'] for group in splits().values() for c in group]
    values += [c[v] for c in cases(10,124) for v in ('base','donor','same')]
    values += [q for c in data(123) for rel in RELATIONS for q in sources(c['p'],rel).values()]
    return {tuple(p[k] for k in SIX_FACTS) for p in values if all(k in p for k in SIX_FACTS)}


def dataset(seed=123, sizes=None):
    sizes = sizes or dict(calibration=10, discovery=30, confirmation=100)
    used_six = old_six_fact_keys()
    used = set()
    groups, streams = [], {}
    for split, n in sizes.items():
        stream = int(fingerprint(['precommit-v1',seed,split])[:16],16)
        streams[split] = stream
        rng = random.Random(stream)
        made = 0
        for _ in range(200000):
            if made == n: break
            p = dict(x=rng.randint(3,6), y=rng.randint(5,8), w=rng.randint(3,8),
                     z=rng.randint(5,8), u=rng.randint(3,6), r=rng.randint(5,8),
                     v=rng.randint(4,6), s=rng.randint(5,8))
            variants = [source(p,rel,k) for rel in RELATIONS for k in SOURCE_KINDS]
            keys = {tuple(q[k] for k in FACTS) for q in variants}
            sixes = {tuple(q[k] for k in SIX_FACTS) for q in variants}
            if keys & used or sixes & used_six: continue
            if any(not 10 <= value <= 99 for rel in RELATIONS for q in variants
                   for value in gold(q,rel).values()): continue
            used.update(keys)
            groups.append(dict(id=f'{split}-{made:03d}', split=split, p=p,
                               instance_sha256=fingerprint(p)))
            made += 1
        if made != n: raise RuntimeError('Could not construct disjoint cases')
    return dict(master_seed=seed, streams=streams, groups=groups,
                historical_exclusion='six-fact variants from all available pilot generators')


def prompt(p, relation, requested='AB', style='symbolic'):
    if requested not in ('AB','BA','A','B') or style not in ('symbolic','numeric'):
        raise ValueError('Unknown prompt variant')
    if relation == 'copy':
        body = f'Displayed number: q = {p["q"]}.\nDefinition of A: A = q.\nDefinition of B: B = q.'
    else:
        a = '(x + y) + (u + r)'
        b = {'independent':'2 * (w + z) + (v + s)',
             'shared':'2 * (x + y) + (v + s)',
             'downstream':'2 * A + (v + s)'}[relation]
        if style == 'numeric':
            for k in FACTS:
                a = re.sub(rf'\b{k}\b',str(p[k]),a)
                b = re.sub(rf'\b{k}\b',str(p[k]),b)
        body = ('Facts: '+'; '.join(f'{k} = {p[k]}' for k in FACTS)+'.\n'
                +f'Definition of A: A = {a}.\nDefinition of B: B = {b}.')
    return ('Use ordinary integer arithmetic.\n'+body+'\n'
            +f'Requested outputs: {", ".join(requested)}.\n'
            +'Fill only the requested final integer values in the supplied answer template. '
            +'No explanation. Do not output intermediate values.')


class CanvasBuilder:
    """Audit 10..99 in full context; support either one or multiple numeral tokens.

    Width is fixed by the domain, never selected using the answer of a problem.
    An unsupported tokenizer is an explicit failure, not a silent filtered sample.
    """
    def __init__(self, backend):
        self.b = backend
        self.cache = {}

    def build(self, p, relation, requested='AB', style='symbolic', unused=False):
        key = (tuple(sorted(p.items())),relation,requested,style,unused)
        if key in self.cache: return self.cache[key]
        b, tok = self.b, self.b.tokenizer
        user = prompt(p,relation,requested,style)
        prefix = tok.apply_chat_template([dict(role='user',content=user)],
                                         tokenize=False,add_generation_prompt=True)
        scaffold = '; '.join(f'{k} = 10' for k in requested)+'.'
        text = prefix+scaffold
        ids = list(tok.encode(text,add_special_tokens=False))
        decoded = [tok.decode(ids[:i],skip_special_tokens=False,
                             clean_up_tokenization_spaces=False) for i in range(len(ids)+1)]
        if decoded[-1] != text or b.mask_id in ids: raise ValueError('Canvas round trip / mask collision')
        spans, mappings = {}, {}
        for m in re.finditer(r'([AB]) = (10)',scaffold):
            field = m[1]; lo = len(prefix)+m.start(2); hi = len(prefix)+m.end(2)
            if text[:lo] not in decoded or text[:hi] not in decoded:
                raise ValueError(f'{field}: numeral boundaries are not token aligned')
            ps = list(range(decoded.index(text[:lo]),decoded.index(text[:hi])))
            mapping = {}
            for value in range(10,100):
                trial = list(tok.encode(text[:lo]+str(value)+text[hi:],add_special_tokens=False))
                if len(trial)!=len(ids) or any(a!=c for i,(a,c) in enumerate(zip(ids,trial)) if i not in ps):
                    raise ValueError(f'{field}: domain-wide tokenization fails at {value}')
                mapping[value] = [trial[i] for i in ps]
            spans[field],mappings[field] = ps,mapping
        if set(spans) != set(requested) or len({len(x) for x in spans.values()}) != 1:
            raise ValueError('Unequal or missing answer widths')
        for ps in spans.values():
            for i in ps: ids[i] = b.mask_id
        neutral = []
        if unused:
            neutral = list(range(len(ids),len(ids)+len(next(iter(spans.values())))))
            ids.extend([b.mask_id]*len(neutral))
        result = dict(ids=ids, spans=spans, maps=mappings, unused=neutral,
                      prompt=user, requested=requested, output_scaffold=scaffold,
                      neutral_policy='trailing unrequested masks held masked; stop after requested answers',
                      uses_gold_to_prepare=False)
        self.cache[key] = result
        return result


def check_alignment(target, src):
    if target['spans'] != src['spans'] or target['unused'] != src['unused'] or len(target['ids']) != len(src['ids']):
        raise ValueError('Source/Target token positions or lengths differ')


def decode(b, cc, state=None, fixed_order=None, patch_context=None):
    ids = list(cc['ids'] if state is None else state)
    answers = [p for ps in cc['spans'].values() for p in ps]
    remaining = [p for p in answers if ids[p]==b.mask_id]
    if fixed_order is not None and (len(fixed_order)!=len(remaining) or set(fixed_order)!=set(remaining)):
        raise ValueError('Fixed order must be a permutation of remaining answer positions')
    writes = []
    while remaining:
        before = ids.copy()
        with ExitStack() as stack:
            active = patch_context is not None and not writes
            if active: stack.enter_context(patch_context())
            preds = b.predict(ids,remaining,[[] for _ in remaining])
        if fixed_order is None:
            index = max(range(len(remaining)),key=lambda j:(preds[j]['confidence'],-remaining[j]))
        else: index = remaining.index(fixed_order[len(writes)])
        pos, pred = remaining[index], preds[index]
        if pred['token_id'] == b.mask_id: raise ValueError('Model committed MASK')
        writes.append(dict(position=pos, token_id=pred['token_id'], confidence=pred['confidence'],
                           state=before, patched=active, predictions=preds, eligible=remaining.copy()))
        ids[pos] = pred['token_id']
        remaining = [p for p in answers if ids[p]==b.mask_id]
    return dict(ids=ids,writes=writes,nfe=len(writes),**read_values(b,ids,cc['spans']))


def layer_grid(depth):
    if depth < 4: raise ValueError('Expected at least four transformer layers')
    return sorted({max(0,round(depth*q)-1) for q in (.25,.5,.75)} | {depth-2,depth-1})


def probes_for(cc, target_gold, alternative):
    positions, probes = [], []
    for field,ps in cc['spans'].items():
        for j,p in enumerate(ps):
            positions.append(p)
            probes.append(sorted({cc['maps'][field][target_gold[field]][j],
                                  cc['maps'][field][alternative[field]][j]}))
    return positions, probes


def log_ratio(cc, predictions, positions, field, target_value, cf_value):
    # Mean of same-state per-token log ratios; NOT a sequence log likelihood.
    by_position = dict(zip(positions,predictions))
    target_ids, cf_ids = cc['maps'][field][target_value],cc['maps'][field][cf_value]
    terms = []
    for j,p in enumerate(cc['spans'][field]):
        probs = by_position[p]['probabilities']
        terms.append(math.log(max(probs[str(cf_ids[j])],1e-38))-math.log(max(probs[str(target_ids[j])],1e-38)))
    return sum(terms)/len(terms)


def scan(b, target, src, direction, expected, alternative, layers):
    """At t=0, both answer regions are mask. Patch physical region positions.

    Dream raw-logit locations are saved separately. Patch sites must not directly
    overlap the destination's readout rows. Include final layer as a null control.
    """
    check_alignment(target,src)
    origin,destination = direction
    positions, probes = probes_for(target,expected,alternative)
    raw_query = raw_logit_positions(target['spans'][destination],b.family)
    assert all(target['ids'][p]==b.mask_id for ps in target['spans'].values() for p in ps)
    assert all(src['ids'][p]==b.mask_id for ps in src['spans'].values() for p in ps)
    base = b.predict(target['ids'],positions,probes)
    modules = b._model.model.layers
    captures = {}
    with ExitStack() as stack:
        for layer in layers:
            captures[layer] = stack.enter_context(ResidualPatch(modules[layer],src['spans'][origin]))
        b.predict(src['ids'],positions,probes)
    baseline_score = log_ratio(target,base,positions,destination,expected[destination],alternative[destination])
    records = []
    for layer in layers:
        for site_name,site in [('answer_'+origin,target['spans'][origin]),('unused',target['unused'])]:
            if not site or len(site)!=len(target['spans'][origin]): raise ValueError('Missing matched unused site')
            if set(site)&set(raw_query): raise ValueError('Patch overlaps destination logit readout')
            with ResidualPatch(modules[layer],site,captures[layer].captured):
                pp = b.predict(target['ids'],positions,probes)
            score = log_ratio(target,pp,positions,destination,expected[destination],alternative[destination])
            records.append(dict(direction=direction,layer=layer+1,site=site_name,positions=site,
                raw_query_positions=raw_query,baseline=base,predictions=pp,
                baseline_log_ratio=baseline_score,log_ratio=score,delta=score-baseline_score,
                state=target['ids'],source_state=src['ids'],event='all_answers_masked_t0'))
    return records


def score_gate(records, styles, accuracy=.8, validity=.95):
    """Require every relation/order/canvas cell to pass; do not select on discovery."""
    scores = {}
    for style in styles:
        cells = {}
        for relation in RELATIONS:
            for requested in ('AB','BA','A','B'):
                for unused in (False,True):
                    rr = [r for r in records if r['style']==style and r['relation']==relation
                          and r['requested']==requested and r['unused']==unused]
                    cells[f'{relation}/{requested}/unused={unused}'] = dict(
                        n=len(rr),accuracy=sum(r['correct'] for r in rr)/len(rr) if rr else 0.,
                        valid=sum(r['valid'] for r in rr)/len(rr) if rr else 0.)
        passing = all(v['n'] and v['accuracy']>=accuracy and v['valid']>=validity for v in cells.values())
        scores[style] = dict(cells=cells,pass_gate=passing,
                             minimum_accuracy=min(v['accuracy'] for v in cells.values()))
    passing_styles = [s for s in styles if scores[s]['pass_gate']]
    chosen = max(passing_styles,key=lambda s:scores[s]['minimum_accuracy']) if passing_styles else None
    return dict(selected_style=chosen,styles=scores,accuracy_threshold=accuracy,validity_threshold=validity)
