"""Exploratory downstream follow-up; frozen v1/v2 implementations stay intact."""
import math
import re
from contextlib import ExitStack

from . import precommit as v1, precommit_v2 as v2
from .backends.base import raw_logit_positions
from .shared_causal import ResidualPatch

SOURCE_KINDS = ('change_A', 'same_A', 'self')
SITES = ('answer_A', 'unused', 'A_to_unused', 'prompt_A')


def source(p, kind, sign):
    if sign not in (-1, 1) or kind not in SOURCE_KINDS:
        raise ValueError((kind, sign))
    q = p.copy()
    if kind != 'self':
        q['x'] += sign
    if kind == 'same_A':
        q['y'] -= sign
    if any(not 1 <= q[k] <= 9 for k in v1.FACTS):
        raise ValueError('Source changes operand token width')
    if any(not 10 <= n <= 99 for n in v2.gold(q, 'downstream').values()):
        raise ValueError('Source leaves audited answer domain')
    return q


class CanvasBuilder:
    """Requested-output wording and physical answer slots are independent factors."""
    def __init__(self, backend):
        self.b, self.cache = backend, {}

    def build(self, p, requested='AB', slots='AB', unused=True):
        if requested not in ('B', 'AB') or slots not in ('B', 'AB'):
            raise ValueError('This experiment crosses B/AB wording with B/AB slots')
        key = (tuple(sorted(p.items())), requested, slots, unused)
        if key in self.cache:
            return self.cache[key]
        b, tok = self.b, self.b.tokenizer
        user = v2.prompt(p, 'downstream', requested, 'numeric')
        prefix = tok.apply_chat_template([dict(role='user', content=user)], tokenize=False,
                                         add_generation_prompt=True)
        suffix = v2.native_suffix(tok, user, prefix)
        scaffold = '; '.join(f'{k} = 10' for k in slots) + '.'
        text = prefix + scaffold + suffix
        ids = list(tok.encode(text, add_special_tokens=False))
        decoded = [tok.decode(ids[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                   for i in range(len(ids)+1)]
        if decoded[-1] != text or b.mask_id in ids:
            raise ValueError('Canvas round trip / mask collision')
        def span(lo, hi):
            if text[:lo] not in decoded or text[:hi] not in decoded:
                raise ValueError('Region boundaries are not token aligned')
            return list(range(decoded.index(text[:lo]), decoded.index(text[:hi])))
        insertion = decoded.index(prefix + scaffold)
        suffix_ids = ids[insertion:].copy()
        spans, mappings = {}, {}
        for m in re.finditer(r'([AB]) = (10)', scaffold):
            field = m[1]
            lo, hi = len(prefix)+m.start(2), len(prefix)+m.end(2)
            positions = span(lo, hi)
            mapping = {}
            for value in range(10, 100):
                trial = list(tok.encode(text[:lo]+str(value)+text[hi:], add_special_tokens=False))
                if len(trial) != len(ids) or any(a != c for i,(a,c) in enumerate(zip(ids,trial)) if i not in positions):
                    raise ValueError(f'{field}: domain-wide tokenization fails at {value}')
                mapping[value] = [trial[i] for i in positions]
            spans[field], mappings[field] = positions, mapping
        if set(spans) != set(slots) or len({len(x) for x in spans.values()}) != 1:
            raise ValueError('Unequal or missing answer widths')
        definition = next(line for line in user.splitlines() if line.startswith('Definition of A:'))
        start = prefix.index(definition)
        # Some BPE tokenizers merge the final period with the following newline.
        # Expand only into whitespace; numeral boundaries above remain strict.
        end = start+len(definition)
        first = max(i for i,s in enumerate(decoded) if len(s)<=start)
        last = min(i for i,s in enumerate(decoded) if len(s)>=end)
        if text[len(decoded[first]):start].strip() or text[end:len(decoded[last])].strip():
            raise ValueError('Prompt region expansion would include neighboring content')
        prompt_A = list(range(first,last))
        prompt_A_text = text[len(decoded[first]):len(decoded[last])]
        for ps in spans.values():
            for pos in ps:
                ids[pos] = b.mask_id
        neutral = []
        if unused:
            width = len(next(iter(spans.values())))
            neutral = list(range(insertion, insertion+width))
            ids[insertion:insertion] = [b.mask_id]*width
        suffix_positions = list(range(insertion+len(neutral), len(ids)))
        assert [ids[i] for i in suffix_positions] == suffix_ids
        cc = dict(ids=ids, spans=spans, maps=mappings, unused=neutral, prompt=user,
                  requested=requested, slots=slots, prompt_A=prompt_A, prompt_A_text=prompt_A_text,
                  output_scaffold=scaffold, ending='native', native_suffix=suffix,
                  suffix_ids=suffix_ids, suffix_positions=suffix_positions,
                  neutral_policy='unrequested masks after scaffold and before native suffix; held masked',
                  task='additive', uses_gold_to_prepare=False)
        self.cache[key] = cc
        return cc


def check_alignment(target, src):
    v1.check_alignment(target, src)
    for key in ('prompt_A', 'suffix_positions', 'suffix_ids', 'requested', 'slots'):
        if target[key] != src[key]:
            raise ValueError('Source/Target alignment differs: '+key)


def prediction_values(b, cc, positions, predictions):
    by_position = dict(zip(positions, predictions))
    texts = {k: b.tokenizer.decode([by_position[pos]['token_id'] for pos in ps])
             for k,ps in cc['spans'].items()}
    values = {k: int(s) if re.fullmatch(r'\d{2}', s) else None for k,s in texts.items()}
    return dict(values=values, texts=texts, valid=all(v is not None for v in values.values()))


def baseline_gate(rows, n=30):
    """Feasibility of the exact patch context, never select individual cases."""
    rr = [r for r in rows if r['requested']=='AB' and r['slots']=='AB' and r['unused']]
    if len(rr) != n or len({r['case'] for r in rr}) != n:
        raise ValueError('Missing or repeated actual patch baselines')
    a = sum(r['t0']['values']['A']==r['expected']['A'] for r in rr)
    b = sum(r['t0']['values']['B']==r['expected']['B'] for r in rr)
    valid = sum(r['t0']['valid'] for r in rr)
    return dict(n=n, t0_A_correct=a, t0_B_correct=b, t0_valid=valid,
                thresholds=dict(A_accuracy=.8, B_accuracy=.5, validity=.95),
                pass_gate=a/n>=.8 and b/n>=.5 and valid/n>=.95,
                context='downstream; requested AB; slots AB; unused True; native ending',
                scope='Exploratory feasibility chosen after v2; old v2 gate unchanged; no case filtering')


def scan(b, target, src, expected, sign, layers, source_kind):
    """One-forward residual-output intervention; no token is committed."""
    check_alignment(target, src)
    if target['slots'] != 'AB' or not target['unused']:
        raise ValueError('Patching requires both slots and matched unused positions')
    for cc in (target, src):
        assert all(cc['ids'][p]==b.mask_id for ps in cc['spans'].values() for p in ps)
        assert all(cc['ids'][p]==b.mask_id for p in cc['unused'])
    alternative = {k: v+sign for k,v in expected.items()}
    positions, probes = v1.probes_for(target, expected, alternative)
    raw_query = raw_logit_positions(target['spans']['B'], b.family)
    regions = dict(answer_A=target['spans']['A'], unused=target['unused'], prompt_A=target['prompt_A'])
    source_regions = dict(answer_A=src['spans']['A'], unused=src['unused'], prompt_A=src['prompt_A'])
    assert len(regions['answer_A']) == len(regions['unused'])
    if any(set(ps) & set(raw_query) for ps in regions.values()):
        raise ValueError('Patch overlaps B raw logit readout')
    base = b.predict(target['ids'], positions, probes)
    modules, captures = b._model.model.layers, {}
    with ExitStack() as stack:
        for layer in layers:
            for region, ps in source_regions.items():
                captures[layer,region] = stack.enter_context(ResidualPatch(modules[layer], ps))
        source_predictions = b.predict(src['ids'], positions, probes)
    baseline_score = v1.log_ratio(target, base, positions, 'B', expected['B'], alternative['B'])
    if not math.isfinite(baseline_score):
        raise ValueError('Non-finite baseline log ratio')
    for layer in layers:
        for site_name in SITES:
            origin = 'answer_A' if site_name=='A_to_unused' else site_name
            destination = 'unused' if site_name=='A_to_unused' else site_name
            tensor = captures[layer,origin].captured
            with ResidualPatch(modules[layer], regions[destination], tensor) as hook:
                pp = b.predict(target['ids'], positions, probes)
            score = v1.log_ratio(target, pp, positions, 'B', expected['B'], alternative['B'])
            delta = score-baseline_score
            if not math.isfinite(delta):
                raise ValueError('Non-finite patch effect')
            if (source_kind=='self' and site_name!='A_to_unused') or layer==len(modules)-1:
                if abs(delta)>1e-6:
                    raise AssertionError('Self/final-layer null failed')
            change_norm = float((tensor.float()-hook.captured.float()).norm())
            original_norm = float(hook.captured.float().norm())
            yield dict(layer=layer+1, site=site_name, source_positions=source_regions[origin],
                target_positions=regions[destination], raw_query_positions=raw_query,
                baseline=base, predictions=pp, source_predictions=source_predictions,
                prediction_positions=positions, baseline_log_ratio=baseline_score,
                log_ratio=score, delta=delta, alternative=alternative,
                B_target_token_ids=target['maps']['B'][expected['B']],
                B_cf_token_ids=target['maps']['B'][alternative['B']],
                B_after=prediction_values(b,target,positions,pp)['values']['B'],
                patch_l2=change_norm, target_l2=original_norm,
                patch_relative_l2=change_norm/max(original_norm,1e-30),
                event='all_answers_masked_t0', direction=['A','B'],
                score='mean per-token log P(B_cf)/P(B_target), same forward; clamp 1e-38')
