"""Predeclared robustness experiments; v1/v2/v2.1 remain frozen."""
import math
import random
from contextlib import ExitStack

from . import precommit as v1, precommit_v2 as v2, precommit_v21 as v21
from .backends.base import raw_logit_positions
from .shared_causal import ResidualPatch

SOURCE_KINDS = ('change_A', 'same_A', 'self', 'A_change_B_fixed', 'A_fixed_B_change')


def source(p, kind, sign):
    if kind not in SOURCE_KINDS or sign not in (-1, 1):
        raise ValueError((kind, sign))
    q = p.copy()
    if kind in ('change_A', 'same_A', 'A_change_B_fixed'):
        q['x'] += sign
    if kind == 'same_A':
        q['y'] -= sign
    if kind == 'A_change_B_fixed':
        q['v'] -= sign
    if kind == 'A_fixed_B_change':
        q['v'] += sign
    if any(not 1 <= q[k] <= 9 for k in v1.FACTS):
        raise ValueError('Source operand width changed')
    for rel in ('downstream', 'independent'):
        if any(not 10 <= x <= 99 for x in v2.gold(q, rel).values()):
            raise ValueError('Source answer outside audited domain')
    return q


def schedule(relation, kind):
    """Layer numbers in returned jobs are one-based; all jobs fixed before inference."""
    if relation not in ('downstream', 'independent') or kind not in SOURCE_KINDS:
        raise ValueError((relation, kind))
    jobs = []
    sites = v21.SITES if relation == 'downstream' else ('answer_A', 'unused')
    for layer in (31, 32):
        for site in sites:
            jobs.append(dict(mode='single', layers=[layer], site=site))
    if relation == 'downstream':
        if kind in ('change_A', 'same_A', 'self'):
            for layer in range(25, 31):
                for site in ('answer_A', 'unused'):
                    jobs.append(dict(mode='single', layers=[layer], site=site))
        for start in (8, 16, 24):
            for site in ('answer_A', 'unused'):
                jobs.append(dict(mode='sustained', layers=list(range(start, 32)), site=site))
    return jobs


def resample_pairs(groups, seed=123, repetitions=5):
    """Five independent uniform permutations conditioned only on no self-pairs."""
    ids = [g['id'] for g in groups]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError('Need at least two unique groups')
    rng = random.Random(seed)
    result = []
    for rep in range(repetitions):
        while True:
            chosen = ids.copy(); rng.shuffle(chosen)
            if all(a != b for a, b in zip(ids, chosen)):
                break
        result.extend(dict(repetition=rep, Target=a, Source=b) for a, b in zip(ids, chosen))
    return result


def audit_splits(data):
    """Check new Source variants plus historical v2 variants across all three splits."""
    pools = {}
    for g in data['groups']:
        keys = v2.variant_keys(g['p'])
        keys |= {tuple(source(g['p'], k, s)[f] for f in v1.FACTS)
                 for k in SOURCE_KINDS for s in (-1, 1)}
        pools.setdefault(g['split'], set()).update(keys)
    intersections = {a+'/'+b: len(pools[a] & pools[b]) for a in pools for b in pools if a < b}
    if pools['calibration'] & pools['discovery']:
        raise ValueError('Exploration/replication Source overlap')
    used = pools['calibration'] | pools['discovery']
    reserved = [g for g in data['groups'] if g['split']=='confirmation']
    target_collisions = [g['id'] for g in reserved if tuple(g['p'][k] for k in v1.FACTS) in used]
    if target_collisions:
        raise ValueError('Reserved Target exposed: '+str(target_collisions))
    family_collisions = []
    for g in reserved:
        variants = v2.variant_keys(g['p']) | {tuple(source(g['p'], k, s)[f] for f in v1.FACTS)
                                           for k in SOURCE_KINDS for s in (-1,1)}
        if variants & used:family_collisions.append(g['id'])
    return dict(unique_variant_counts={k:len(v) for k,v in pools.items()}, intersections=intersections,
        confirmation_target_collisions=target_collisions, confirmation_family_collisions=family_collisions,
        reservation_note='No confirmation Target evaluated. Replace flagged families before future independent confirmation with these Source rules.')


def specification(seed=123):
    return dict(protocol='precommit-v2.2', seed=seed, splits=['calibration', 'discovery'],
        relation_sources={'downstream':list(SOURCE_KINDS), 'independent':list(SOURCE_KINDS[:3])},
        schedules={r:{k:schedule(r,k) for k in ks} for r,ks in
                   [('downstream',SOURCE_KINDS), ('independent',SOURCE_KINDS[:3])]},
        resample=dict(repetitions=5, seed=seed, layer=31, sites=['answer_A','unused'],
                      selection='uniform derangements; no filtering on A, B, or accuracy'),
        primary_context='AB wording, AB slots, unused=True, native ending, all masked t=0',
        sustained='each layer gets the corresponding clean Source block output; no cross-forward state',
        replication='same entire grid on reserved discovery30, no best-layer selection; confirmation100 untouched',
        gate='report baseline only; no filtering of cases or discovery runs by accuracy',
        primary_contrasts=['L31 A_change_B_fixed minus A_fixed_B_change at answer_A',
                           'L31 change_A minus same_A at answer_A',
                           'downstream minus independent L31 change_A-minus-same_A at answer_A'],
        metric='mean same-forward token log P(B+sign)/P(B), then minus baseline; independent candidate is off-graph',
        bootstrap='4000 resamples of base groups, seed123; signs and resample choices averaged within group')


class CanvasBuilder:
    """Use frozen v2 canvas with an audited Definition-of-A span."""
    def __init__(self, b):
        self.b, self.old, self.cache = b, v2.CanvasBuilder(b), {}

    def build(self, p, relation):
        key = (tuple(sorted(p.items())), relation)
        if key in self.cache:
            return self.cache[key]
        cc = dict(self.old.build(p, relation, 'AB', unused=True))
        tok = self.b.tokenizer
        prefix = tok.apply_chat_template([dict(role='user', content=cc['prompt'])],
                                        tokenize=False, add_generation_prompt=True)
        definition = next(s for s in cc['prompt'].splitlines() if s.startswith('Definition of A:'))
        lo = prefix.index(definition); hi = lo+len(definition)
        ids = tok.encode(prefix, add_special_tokens=False)
        decoded = [tok.decode(ids[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                   for i in range(len(ids)+1)]
        first = max(i for i,s in enumerate(decoded) if len(s)<=lo)
        last = min(i for i,s in enumerate(decoded) if len(s)>=hi)
        if prefix[len(decoded[first]):lo].strip() or prefix[hi:len(decoded[last])].strip():
            raise ValueError('Prompt region includes neighboring content')
        cc.update(slots='AB', prompt_A=list(range(first,last)),
                  prompt_A_text=prefix[len(decoded[first]):len(decoded[last])], relation=relation)
        actual = tok.decode([cc['ids'][p] for p in cc['prompt_A']],
                            skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if actual != cc['prompt_A_text']:
            raise ValueError('Prompt region changed at chat prefix boundary')
        self.cache[key] = cc
        return cc


def evaluate(b, target, src, expected, alternative, jobs, source_kind):
    """Return context + rows, retaining probabilities but never storing GPU tensors."""
    v21.check_alignment(target, src)
    for cc in (target, src):
        assert all(cc['ids'][p]==b.mask_id for ps in cc['spans'].values() for p in ps)
        assert all(cc['ids'][p]==b.mask_id for p in cc['unused'])
    positions, probes = v1.probes_for(target, expected, alternative)
    raw_query = raw_logit_positions(target['spans']['B'], b.family)
    regions = {'answer_A':target['spans']['A'], 'unused':target['unused'], 'prompt_A':target['prompt_A']}
    source_regions = {'answer_A':src['spans']['A'], 'unused':src['unused'], 'prompt_A':src['prompt_A']}
    assert len(regions['answer_A'])==len(regions['unused'])
    assert not any(set(ps)&set(raw_query) for ps in regions.values())
    modules = b._model.model.layers
    needed = {(l-1, 'answer_A' if j['site']=='A_to_unused' else j['site'])
              for j in jobs for l in j['layers']}
    if any(l<0 or l>=len(modules) for l,_ in needed):
        raise ValueError('Layer outside model')
    base = b.predict(target['ids'], positions, probes)
    captures = {}
    with ExitStack() as stack:
        for l,site in sorted(needed):
            captures[l,site] = stack.enter_context(ResidualPatch(modules[l], source_regions[site]))
        source_pred = b.predict(src['ids'], positions, probes)
    baseline = v1.log_ratio(target, base, positions, 'B', expected['B'], alternative['B'])
    context = dict(baseline=base, source_predictions=source_pred,
        baseline_values=v21.prediction_values(b,target,positions,base)['values'],
        source_values=v21.prediction_values(b,src,positions,source_pred)['values'],
        expected=expected, alternative=alternative, baseline_log_ratio=baseline,
        prediction_positions=positions, raw_query_positions=raw_query,
        B_target_token_ids=target['maps']['B'][expected['B']],
        B_candidate_token_ids=target['maps']['B'][alternative['B']])
    rows = []
    for job in jobs:
        origin = 'answer_A' if job['site']=='A_to_unused' else job['site']
        destination = 'unused' if job['site']=='A_to_unused' else job['site']
        hooks = []
        with ExitStack() as stack:
            for l in job['layers']:
                hooks.append(stack.enter_context(ResidualPatch(modules[l-1], regions[destination],
                                                               captures[l-1,origin].captured)))
            pred = b.predict(target['ids'], positions, probes)
        score = v1.log_ratio(target, pred, positions, 'B', expected['B'], alternative['B'])
        delta = score-baseline
        assert math.isfinite(delta)
        if (source_kind=='self' and job['site']!='A_to_unused') or job['layers']==[len(modules)]:
            if abs(delta)>1e-6:
                raise AssertionError('Self/last-layer null failed')
        norms = []
        for l,hook in zip(job['layers'],hooks):
            tensor = captures[l-1,origin].captured
            norm = float(hook.captured.float().norm())
            change = float((tensor.float()-hook.captured.float()).norm())
            norms.append(dict(layer=l, patch_l2=change, target_l2=norm, relative=change/max(norm,1e-30)))
        rows.append(dict(**job, source_positions=source_regions[origin], target_positions=regions[destination],
            predictions=pred, values=v21.prediction_values(b,target,positions,pred)['values'],
            log_ratio=score, delta=delta, norms=norms, event='all_answers_masked_t0'))
    return context, rows
