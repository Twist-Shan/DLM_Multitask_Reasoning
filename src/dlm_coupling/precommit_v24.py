"""Exploratory dependency selection, spatial order, and canvas capacity.

v2.4 uses the previously inspected calibration30, not the confirmation split.
Layer numbers are one-based block outputs; Source supplies, Target receives.
"""
from contextlib import ExitStack
import math
import random
import re

from . import precommit as v1, precommit_v2 as v2
from .backends.base import raw_logit_positions
from .shared_causal import ResidualPatch, read_values

PROTOCOL = 'precommit-v2.4'
LAYOUTS = ('ABC', 'CBA')
MODES = {'L30': [30], 'L31': [31], 'L32': [32], 'L24_31': list(range(24, 32))}


def gold(p, parent):
    if parent not in ('A', 'C'):
        raise ValueError(parent)
    values = {k: sum(p[k.lower()+str(j)] for j in range(1, 5)) for k in ('A', 'C')}
    return dict(**values, B=values[parent]+p['v1']+p['v2'])


def source(p, parent, kind, sign):
    if sign not in (-1, 1):
        raise ValueError(sign)
    q = p.copy()
    if kind == 'B_only':
        q['v1'] += sign
    elif kind != 'self':
        match = re.fullmatch(r'(change|same|comp)_([AC])', kind)
        if not match:
            raise ValueError(kind)
        operation, variable = match.groups()
        q[variable.lower()+'1'] += sign
        if operation == 'same':
            q[variable.lower()+'2'] -= sign
        if operation == 'comp' and variable == parent:
            q['v1'] -= sign
    if any(not 1 <= x <= 9 for x in q.values()):
        raise ValueError('Source changed operand width')
    if any(not 10 <= x <= 99 for x in gold(q, parent).values()):
        raise ValueError('Source left audited answer domain')
    return q


def dataset(seed=123):
    old = v2.dataset(seed)
    base = [g for g in old['groups'] if g['split'] == 'calibration']
    stream = int(v1.fingerprint([PROTOCOL, seed, 'independent-C'])[:16], 16)
    rng = random.Random(stream)
    equal_ids = set(rng.sample(range(len(base)), len(base)//2))
    groups, seen = [], set()
    for i, g in enumerate(base):
        oldp = g['p']
        p = dict(zip(('a1','a2','a3','a4','v1','v2'),
                     (oldp['x'],oldp['y'],oldp['u'],oldp['r'],oldp['v'],oldp['s'])))
        a = sum(p[f'a{j}'] for j in range(1,5))
        for _ in range(100000):
            c = [rng.randint(3,8) for _ in range(4)]
            if (sum(c) == a) != (i in equal_ids):
                continue
            # Do not make the equal-value branch a literal operand duplicate.
            if c == [p[f'a{j}'] for j in range(1,5)]:
                continue
            q = dict(p, **{f'c{j+1}':x for j,x in enumerate(c)})
            key = tuple(sorted(q.items()))
            if key in seen:
                continue
            seen.add(key);p=q;break
        else:
            raise RuntimeError('C generation failed')
        groups.append(dict(id=f'v24-exploration-{i:03d}', base_case=g['id'], p=p,
                           equal_AC=i in equal_ids, instance_sha256=v1.fingerprint(p)))
    return dict(protocol=PROTOCOL, seed=seed, stream=stream, groups=groups,
                historical_dataset_sha256=v1.fingerprint(old),
                scope='A and V reused from inspected v2 calibration30; C newly sampled; no confirmation inference')


def prompt(p, parent):
    gold(p, parent)  # Reject an undefined dependency.
    return ('Solve the arithmetic exactly.\n'
            f"Definition of A: A = ({p['a1']} + {p['a2']}) + ({p['a3']} + {p['a4']}).\n"
            f"Definition of C: C = ({p['c1']} + {p['c2']}) + ({p['c3']} + {p['c4']}).\n"
            f"Definition of B: B = {parent} + ({p['v1']} + {p['v2']}).\n"
            'Fill the labelled answer fields in the supplied response template. '
            'Each answer is a two-digit integer. Do not write intermediate steps or extra text.')


class CanvasBuilder:
    def __init__(self, backend):
        self.b, self.cache = backend, {}

    def build(self, p, parent, layout='ABC', extra=2, filler='mask'):
        if layout not in ('ABC','CBA','B','BA','BC') or extra < 0 or filler not in ('mask','eos'):
            raise ValueError((layout,extra,filler))
        key = (tuple(sorted(p.items())),parent,layout,extra,filler)
        if key in self.cache:
            return self.cache[key]
        b, tok = self.b, self.b.tokenizer
        user = prompt(p,parent)
        prefix = tok.apply_chat_template([dict(role='user',content=user)], tokenize=False, add_generation_prompt=True)
        suffix = v2.native_suffix(tok,user,prefix)
        # A fixed period after each field preserves the B prefix across space variants.
        scaffold = ' '.join(f'{k} = 10.' for k in layout)
        text = prefix+scaffold+suffix
        ids = list(tok.encode(text,add_special_tokens=False))
        decoded = [tok.decode(ids[:i],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                   for i in range(len(ids)+1)]
        if decoded[-1] != text or b.mask_id in ids:
            raise ValueError('Canvas round trip or mask collision')
        def boundary(n):
            try:
                return decoded.index(text[:n])
            except ValueError as exc:
                raise ValueError('Unaligned scaffold boundary') from exc
        insertion = boundary(len(prefix+scaffold))
        prompt_end = boundary(len(prefix))
        spans, maps = {}, {}
        for m in re.finditer(r'([ABC]) = (10)',scaffold):
            lo,hi = len(prefix)+m.start(2),len(prefix)+m.end(2)
            ps = list(range(boundary(lo),boundary(hi)))
            if len(ps) != 2:
                raise ValueError('This protocol requires two digit-token positions')
            mapping = {}
            for value in range(10,100):
                trial = list(tok.encode(text[:lo]+str(value)+text[hi:],add_special_tokens=False))
                if len(trial)!=len(ids) or any(a!=c for i,(a,c) in enumerate(zip(ids,trial)) if i not in ps):
                    raise ValueError(f'Unstable numeral tokenization: {m[1]}={value}')
                mapping[value] = [trial[i] for i in ps]
            spans[m[1]],maps[m[1]] = ps,mapping
        if set(spans) != set(layout):
            raise ValueError('Missing or duplicate fields')
        suffix_ids = ids[insertion:].copy()
        for ps in spans.values():
            for pos in ps: ids[pos] = b.mask_id
        fill_id = b.mask_id if filler=='mask' else tok.eos_token_id
        if not isinstance(fill_id,int) or (filler=='eos' and fill_id==b.mask_id):
            raise ValueError('Tokenizer has no distinct EOS ID')
        ids[insertion:insertion] = [fill_id]*extra
        cc = dict(ids=ids,spans=spans,maps=maps,mask_id=b.mask_id,prompt=user,parent=parent,layout=layout,
                  output_scaffold=scaffold,extra=extra,filler=filler,
                  unused=list(range(insertion,insertion+extra)),
                  suffix_positions=list(range(insertion+extra,len(ids))),suffix_ids=suffix_ids,
                  native_suffix=suffix,prompt_length=prompt_end,uses_gold_to_prepare=False)
        assert [ids[i] for i in cc['suffix_positions']]==suffix_ids
        self.cache[key]=cc
        return cc

    def space(self,p,parent,condition):
        if condition.startswith(('mask_','eos_')):
            filler,count = condition.split('_')
            cc = dict(self.build(p,parent,'B',int(count),filler))
            cc['patch_positions'] = cc['unused'][:2]
        elif condition in ('slot_A','slot_C','matched_mask','matched_eos'):
            a,c,base = (self.build(p,parent,layout,0) for layout in ('BA','BC','B'))
            if len(a['ids'])!=len(c['ids']) or a['spans']['A']!=c['spans']['C']:
                raise ValueError('A/C tail slots have unequal token geometry')
            extra = len(a['ids'])-len(base['ids'])
            if condition=='slot_A': cc=dict(a)
            elif condition=='slot_C': cc=dict(c)
            else: cc=dict(self.build(p,parent,'B',extra,condition.removeprefix('matched_')))
            cc['patch_positions'] = a['spans']['A']
            if cc['spans']['B']!=base['spans']['B'] or len(cc['ids'])!=len(a['ids']):
                raise ValueError('Space comparison is not position/length matched')
            if condition.startswith('matched_') and not set(cc['patch_positions'])<=set(cc['unused']):
                raise ValueError('Matched digit offsets outside neutral tail')
        else:
            raise ValueError(condition)
        cc['condition']=condition
        return cc


def check_alignment(target,src):
    for key in ('spans','suffix_positions','prompt_length','unused'):
        if target[key]!=src[key]: raise ValueError('Source/Target misalignment: '+key)
    if len(target['ids'])!=len(src['ids']): raise ValueError('Source/Target lengths differ')
    n=target['prompt_length']
    if target['ids'][n:]!=src['ids'][n:]: raise ValueError('Source/Target visible canvas differs')


def read_prediction(b,cc,positions,predictions):
    ids=cc['ids'].copy()
    for pos,pred in zip(positions,predictions): ids[pos]=pred['token_id']
    return read_values(b,ids,{k:ps for k,ps in cc['spans'].items() if set(ps)<=set(positions)})


def schedule(kind):
    jobs=[]
    def add(site,modes):
        for mode in modes: jobs.append(dict(site=site,mode=mode,layers=MODES[mode]))
    if kind=='self':
        for site in ('A','C'):add(site,('L31','L24_31'))
        add('unused',('L31',))
    elif kind=='B_only':
        for site in ('A','C'):add(site,('L31',))
    else:
        operation,variable=kind.split('_')
        add(variable,('L30','L31','L32','L24_31') if operation=='change' else ('L31','L24_31'))
        if operation=='change':add('unused',('L31','L24_31'))
    return jobs


def patch_scan(b,target,src,expected,sign,jobs,kind,regions=None):
    """Only single-forward predictions; no values are inserted into the input."""
    check_alignment(target,src)
    for cc in (target,src):
        if any(cc['ids'][p]!=b.mask_id for ps in cc['spans'].values() for p in ps):
            raise ValueError('t=0 scan contains committed answer tokens')
    alternative=dict(expected,B=expected['B']+sign)
    positions,probes=v1.probes_for(target,expected,alternative)
    raw_query=raw_logit_positions(target['spans']['B'],b.family)
    regions=regions or dict(A=target['spans']['A'],C=target['spans']['C'],unused=target['unused'])
    if any(not ps or set(ps)&set(raw_query) for ps in regions.values()):
        raise ValueError('Invalid or direct B-readout patch site')
    modules=b._model.model.layers
    needed={(l,job['site']) for job in jobs for l in job['layers']}
    base=b.predict(target['ids'],positions,probes)
    baseline=v1.log_ratio(target,base,positions,'B',expected['B'],alternative['B'])
    captures={}
    with ExitStack() as stack:
        for layer,site in sorted(needed):
            if not 1<=layer<=len(modules):raise ValueError('Layer outside model')
            captures[layer,site]=stack.enter_context(ResidualPatch(modules[layer-1],regions[site]))
        source_pred=b.predict(src['ids'],positions,probes)
    rows=[]
    for job in jobs:
        with ExitStack() as stack:
            hooks=[stack.enter_context(ResidualPatch(modules[l-1],regions[job['site']],
                         captures[l,job['site']].captured)) for l in job['layers']]
            pred=b.predict(target['ids'],positions,probes)
        score=v1.log_ratio(target,pred,positions,'B',expected['B'],alternative['B'])
        delta=score-baseline
        if not math.isfinite(delta):raise ValueError('Nonfinite effect')
        if (kind=='self' or job['layers']==[len(modules)]) and abs(delta)>1e-6:
            raise AssertionError('Self or final-layer cross-position null failed')
        norms=[float((captures[l,job['site']].captured.float()-h.captured.float()).norm())
               for l,h in zip(job['layers'],hooks)]
        rows.append(dict(**job,delta=delta,log_ratio=score,predictions=pred,
                         after=read_prediction(b,target,positions,pred),patch_l2=norms,
                         patch_positions=regions[job['site']],raw_query_positions=raw_query))
    context=dict(expected=expected,alternative=alternative,positions=positions,baseline=base,
                 baseline_values=read_prediction(b,target,positions,base),source_predictions=source_pred,
                 source_values=read_prediction(b,src,positions,source_pred),baseline_log_ratio=baseline,
                 B_target_token_ids=target['maps']['B'][expected['B']],
                 B_candidate_token_ids=target['maps']['B'][alternative['B']])
    return context,rows


def decode(b,cc,policy='B_first',source_canvas=None,patch_mode='none',site=None,layers=None,only_B=False):
    """Hold A/C masked until both B digits commit; refresh Source each forward.

    For persistent interventions, Source sees the current Target visible canvas.
    This can disagree with Source facts and is recorded, never oracle-completed.
    """
    if policy not in ('B_first','confidence') or patch_mode not in ('none','first','persistent'):
        raise ValueError((policy,patch_mode))
    if patch_mode!='none':
        if source_canvas is None or not site or not layers:raise ValueError('Incomplete intervention')
        check_alignment(cc,source_canvas)
        if set(site)&set(cc['spans']['B']):raise ValueError('Direct B patch forbidden')
    ids=cc['ids'].copy();writes=[]
    active_fields=['B'] if only_B else list(cc['spans'])
    answers=[p for k in active_fields for p in cc['spans'][k]]
    while any(ids[p]==b.mask_id for p in answers):
        remaining=[p for p in answers if ids[p]==b.mask_id]
        missing_B=[p for p in cc['spans']['B'] if ids[p]==b.mask_id]
        eligible=missing_B if policy=='B_first' and missing_B else remaining
        active=bool(missing_B) and (patch_mode=='persistent' or (patch_mode=='first' and not writes))
        source_state=None;captures={}
        if active:
            if any(ids[p]!=b.mask_id for p in site):raise AssertionError('Patched answer site already committed')
            source_state=source_canvas['ids'].copy()
            n=cc['prompt_length'];source_state[n:]=ids[n:]
            with ExitStack() as stack:
                for l in layers:captures[l]=stack.enter_context(ResidualPatch(b._model.model.layers[l-1],site))
                b.predict(source_state,eligible,[[] for _ in eligible])
        before=ids.copy()
        with ExitStack() as stack:
            if active:
                for l in layers:stack.enter_context(ResidualPatch(b._model.model.layers[l-1],site,captures[l].captured))
            predictions=b.predict(ids,eligible,[[] for _ in eligible])
        index=max(range(len(eligible)),key=lambda j:(predictions[j]['confidence'],-eligible[j]))
        pos,pred=eligible[index],predictions[index]
        if pred['token_id']==b.mask_id:raise ValueError('Committed MASK')
        writes.append(dict(position=pos,state=before,eligible=eligible,predictions=predictions,
                           patched=active,Source_state=source_state,token_id=pred['token_id']))
        ids[pos]=pred['token_id']
    if any(ids[p]!=cc['ids'][p] for p in range(len(ids)) if p not in answers):
        raise AssertionError('Decoder modified unrequested positions')
    return dict(ids=ids,writes=writes,nfe=len(writes),policy=policy,patch_mode=patch_mode,
                **read_values(b,ids,{k:cc['spans'][k] for k in active_fields}))


def specification(seed=123):
    return dict(protocol=PROTOCOL,seed=seed,model='illada',groups=30,equal_AC=15,
        parents=['A','C'],layouts=list(LAYOUTS),layer_modes=MODES,
        kinds=['change_A','same_A','comp_A','change_C','same_C','comp_C','B_only','self'],
        space_quantity=['mask_0','mask_2','eos_2','mask_8','eos_8','mask_32','eos_32'],
        space_usage=['slot_A','slot_C','matched_mask','matched_eos'],
        metric='mean per-digit same-forward log P(B+sign)/P(B), minus clean Target; not joint likelihood',
        primary='L31 dependency-selective effect: relevant (change-same) minus irrelevant (change-same), paired within group',
        controls='compensation holds Source B fixed; B_only; unused; self; last-layer null; no case filtering',
        rollouts='B-first baseline vs first-only/persistent 24..31; fresh Source with current Target visible tokens; finish all fields',
        space='Only B commits (2 forwards); B positions and prompt fixed; matched usage conditions also total length/suffix offsets fixed',
        uncertainty='Exploratory; 4000 bootstrap resamples of 30 base groups, seed123; no confirmatory p values',
        confirmation='No confirmation or discovery groups evaluated')
