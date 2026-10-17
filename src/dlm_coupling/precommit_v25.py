"""Return to the exact v2 tasks; vary order/space separately from redundancy.

Old C=x+y retains its meaning. R is a separately defined duplicate expression,
not an alias R=A: Source may change either expression independently.
"""
import re

from . import precommit as v1, precommit_v2 as v2, precommit_v22 as v22
from . import precommit_v24 as mechanics

PROTOCOL = 'precommit-v2.5'
MODES = mechanics.MODES
QUANTITIES = ('mask_0', 'mask_2', 'mask_8', 'mask_32', 'eos_2', 'eos_8', 'eos_32')
ALIAS_KINDS = ('change_A','same_A','comp_A','change_R','same_R','comp_R','B_only','self')


def dataset(seed=123):
    old = v2.dataset(seed)
    groups = [g for g in old['groups'] if g['split']=='calibration']
    return dict(protocol=PROTOCOL,seed=seed,groups=groups,
                historical_dataset_sha256=v1.fingerprint(old),
                scope='Exact previously inspected v2 calibration30; no new problems or confirmation inference')


def redundant(p):
    return dict(p=p.copy(), R=[p[k] for k in ('x','y','u','r')])


def redundant_gold(item,parent):
    if parent not in ('A','R'):raise ValueError(parent)
    p=item['p'];a=v2.gold(p,'downstream')['A'];r=sum(item['R'])
    return dict(A=a,R=r,B=(a if parent=='A' else r)+p['v']+p['s'])


def redundant_source(item,parent,kind,sign):
    if kind not in ALIAS_KINDS or sign not in (-1,1):raise ValueError((kind,sign))
    redundant_gold(item,parent)
    q=dict(p=item['p'].copy(),R=item['R'].copy())
    if kind=='B_only':q['p']['v']+=sign
    elif kind!='self':
        operation,variable=kind.split('_')
        if variable=='A':
            q['p']['x']+=sign
            if operation=='same':q['p']['y']-=sign
        else:
            q['R'][0]+=sign
            if operation=='same':q['R'][1]-=sign
        if operation=='comp' and variable==parent:q['p']['v']-=sign
    if any(not 1<=x<=9 for x in [*q['p'].values(),*q['R']]):raise ValueError('Operand width changed')
    if any(not 10<=x<=99 for x in redundant_gold(q,parent).values()):raise ValueError('Answer domain')
    return q


def redundant_prompt(item,parent,request_R=True):
    redundant_gold(item,parent)
    text=v2.prompt(item['p'],'downstream','AB','numeric')
    r=item['R']
    definition=f'Definition of R: R = ({r[0]} + {r[1]}) + ({r[2]} + {r[3]}).\n'
    text=text.replace('Definition of B:',definition+'Definition of B:')
    if parent=='R':text=text.replace('B = A +','B = R +')
    if request_R:text=text.replace('Requested outputs: A, B.','Requested outputs: A, B, R.')
    return text


class CanvasBuilder:
    """Old numeric prompt, semicolon scaffold and native ending, audited in context."""
    def __init__(self,b):self.b,self.cache=b,{}

    def core(self,p,relation,layout='AB',quantity='mask_2'):
        if layout not in ('AB','BA') or quantity not in QUANTITIES:raise ValueError((layout,quantity))
        filler,extra=quantity.split('_')
        cc=self.build(v2.prompt(p,relation,'AB','numeric'),layout,int(extra),filler)
        return dict(cc,relation=relation,requested='AB',condition=quantity)

    def alias(self,item,parent,layout='ABR'):
        if layout not in ('AB','ABR','RBA'):raise ValueError(layout)
        cc=self.build(redundant_prompt(item,parent,layout!='AB'),layout,2,'mask')
        return dict(cc,parent=parent,requested='AB' if layout=='AB' else 'ABR')

    def build(self,user,layout,extra,filler):
        key=(user,layout,extra,filler)
        if key in self.cache:return self.cache[key]
        b,tok=self.b,self.b.tokenizer
        prefix=tok.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True)
        suffix=v2.native_suffix(tok,user,prefix)
        scaffold='; '.join(f'{k} = 10' for k in layout)+'.'
        text=prefix+scaffold+suffix
        ids=list(tok.encode(text,add_special_tokens=False))
        decoded=[tok.decode(ids[:i],skip_special_tokens=False,clean_up_tokenization_spaces=False) for i in range(len(ids)+1)]
        if decoded[-1]!=text or b.mask_id in ids:raise ValueError('Round trip or mask collision')
        def boundary(n):
            if text[:n] not in decoded:raise ValueError('Unaligned boundary')
            return decoded.index(text[:n])
        prompt_end=boundary(len(prefix));insertion=boundary(len(prefix+scaffold))
        spans,maps={},{}
        for match in re.finditer(r'([ABR]) = (10)',scaffold):
            lo,hi=len(prefix)+match.start(2),len(prefix)+match.end(2)
            ps=list(range(boundary(lo),boundary(hi)))
            if len(ps)!=2:raise ValueError('Requires two digit-token positions')
            mapping={}
            for value in range(10,100):
                trial=list(tok.encode(text[:lo]+str(value)+text[hi:],add_special_tokens=False))
                if len(trial)!=len(ids) or any(a!=c for i,(a,c) in enumerate(zip(ids,trial)) if i not in ps):
                    raise ValueError('Full-context 10..99 token audit failed')
                mapping[value]=[trial[i] for i in ps]
            spans[match[1]],maps[match[1]]=ps,mapping
        if set(spans)!=set(layout):raise ValueError('Missing field')
        suffix_ids=ids[insertion:].copy()
        for ps in spans.values():
            for pos in ps:ids[pos]=b.mask_id
        fill=b.mask_id if filler=='mask' else tok.eos_token_id
        if not isinstance(fill,int) or (filler=='eos' and fill==b.mask_id):raise ValueError('Invalid EOS')
        ids[insertion:insertion]=[fill]*extra
        cc=dict(ids=ids,spans=spans,maps=maps,mask_id=b.mask_id,prompt=user,layout=layout,
                output_scaffold=scaffold,extra=extra,filler=filler,unused=list(range(insertion,insertion+extra)),
                suffix_positions=list(range(insertion+extra,len(ids))),suffix_ids=suffix_ids,
                native_suffix=suffix,prompt_length=prompt_end,uses_gold_to_prepare=False)
        self.cache[key]=cc
        return cc


def core_kinds(relation):
    return ('change_A','same_A','self','A_change_B_fixed','A_fixed_B_change') if relation=='downstream' else ('change_A','same_A','self')


def core_schedule(kind):
    modes=('L30','L31','L32','L24_31') if kind=='change_A' else ('L31','L24_31')
    if kind in ('self','A_fixed_B_change'):modes=('L31',)
    return [dict(site=site,mode=mode,layers=MODES[mode]) for site in ('A','unused') for mode in modes]


def alias_schedule(kind):
    # Identical layer/control coverage to v2.4; only the task and prompt change.
    jobs=mechanics.schedule(kind.replace('_R','_C'))
    return [dict(j,site='R' if j['site']=='C' else j['site']) for j in jobs]


def core_conditions():
    return [(rel,layout,quantity) for rel in v2.RELATIONS for layout in ('AB','BA')
            for quantity in (QUANTITIES if rel=='downstream' else ('mask_2',))]


def specification(seed=123):
    return dict(protocol=PROTOCOL,seed=seed,groups=30,model='illada',core_conditions=core_conditions(),
        alias_parents=['A','R'],alias_layouts=['AB','ABR','RBA'],alias_kinds=list(ALIAS_KINDS),
        layer_modes=MODES,policies=['confidence','B_first'],
        original_anchor='Exact v2 numeric AB prompt, AB scaffold, two tail masks, native ending; audit token equality',
        redundancy='Target R repeats the numeric A expression; independently intervenable definitions, not R=A',
        spatial='Core AB/BA keeps request AB fixed; absolute B position changes. Alias ABR/RBA keeps B position fixed.',
        space='Same original question and AB/BA answers; only tail count/type changes; four commits in every condition',
        inference='Existing calibration30, no correctness selection; exploratory group-clustered bootstrap seed123',
        planned_per_group=dict(core_behavior=36,core_contexts=28,core_patch=128,core_rollout=8,
                               alias_behavior=12,alias_contexts=60,alias_patch=196))


check_alignment=mechanics.check_alignment
patch_scan=mechanics.patch_scan
decode=mechanics.decode
read_prediction=mechanics.read_prediction
core_source=v22.source
