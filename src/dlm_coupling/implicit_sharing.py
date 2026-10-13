"""Answer-only arithmetic: source-variable/answer dissociation, seed 123.

No intermediate output slots or forced task order. Private research pilot.
"""
import random
import re
from contextlib import ExitStack
from .shared_causal import ResidualPatch, read_values
from .headbank_pilot import BankPatch

RELATIONS = ('shared', 'independent', 'downstream')
SOURCE_KINDS = ('same_value', 'same_C_diff_answers', 'diff_C_same_answers',
                'diff_C_diff_answers', 'C_only', 'U_only', 'V_only')


def gold(p, relation='shared', c=None):
    c = p['x'] + p['y'] if c is None else c
    u, v = p['u'] + p['r'], p['v'] + p['s']
    a = c + u
    b = 2*c+v if relation == 'shared' else 2*v if relation == 'independent' else 2*a+v
    return dict(A=a, B=b)


def sources(p, relation):
    # All inputs remain one digit; compensating changes preserve BOTH answers.
    return dict(self=p.copy(), same_value=dict(p, x=p['x']-1, y=p['y']+1),
        same_C_diff_answers=dict(p, u=p['u']+2, v=p['v']+3),
        diff_C_same_answers=dict(p, x=p['x']+1, u=p['u']-1,
                                v=p['v']-(2 if relation == 'shared' else 0)),
        diff_C_diff_answers=dict(p, x=p['x']+1, u=p['u']+2, v=p['v']+3),
        C_only=dict(p, x=p['x']+1), U_only=dict(p, u=p['u']+1),
        V_only=dict(p, v=p['v']+1))


def data(seed=123):
    # Exclude previous pilot variants, including earlier validation groups.
    from .mechanism_pilot import dataset
    from .headbank_pilot import validation_cases
    from .baseline_repair import splits
    from .shared_followup import cases
    key = lambda p: tuple(sorted(p.items()))
    used = {key(p) for c in dataset()+validation_cases() for p in c['variants'].values()}
    used.update(key(c['base']) for group in splits().values() for c in group)
    used.update(key(c[v]) for c in cases(10,124) for v in ('base','donor','same'))
    rng, result = random.Random(seed), []
    for split, n in [('calibration',10), ('discovery',4), ('validation',10)]:
        count=0
        for _ in range(100000):
            if count==n: break
            p = dict(x=rng.randint(3,6),y=rng.randint(5,8),u=rng.randint(3,6),
                     r=rng.randint(6,8),v=rng.randint(4,6),s=rng.randint(5,7))
            variants = {key(q) for rel in RELATIONS for q in sources(p,rel).values()}
            if variants & used: continue
            used.update(variants)
            result.append(dict(id=f'{split}-{count}', split=split, p=p)); count+=1
        if count != n: raise RuntimeError('Could not construct disjoint groups')
    return result


def prompt(p, relation, requested='AB', style='symbolic'):
    if relation not in RELATIONS or requested not in ('A','B','AB','BA'):
        raise ValueError('Unknown task condition')
    a='(x + y) + (u + r)'
    b={'shared':'2 * (x + y) + (v + s)', 'independent':'2 * (v + s)',
       'downstream':'2 * A + (v + s)'}[relation]
    if style=='expanded' and relation=='shared': b='2 * x + 2 * y + (v + s)'
    elif style not in ('symbolic','expanded','compact','numeric'): raise ValueError(style)
    facts='; '.join(f'{k} = {v}' for k,v in p.items())
    if style=='numeric':
        for k,v in p.items():
            a=re.sub(rf'\b{k}\b',str(v),a); b=re.sub(rf'\b{k}\b',str(v),b)
    if style=='compact':
        return (f'Facts: {facts}.\nDefinition of A: A = {a}.\nDefinition of B: B = {b}.\n'
                f'Compute {" and ".join(requested)}. Fill only the final integer values. No explanation.')
    return (f'Use ordinary integer arithmetic.\nFacts: {facts}.\n'
            f'Definition of A: A = {a}.\nDefinition of B: B = {b}.\n'
            f'Requested outputs: {", ".join(requested)}.\n'
            'Fill only the requested final integer values in the supplied answer template. '
            'No explanation. Do not output intermediate values.')


def canvas(b,p,relation,requested='AB',style='symbolic'):
    user=prompt(p,relation,requested,style); tok=b.tokenizer
    prefix=tok.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True)
    text=prefix+'; '.join(f'{k} = 00' for k in requested)+'.'
    ids=tok.encode(text,add_special_tokens=False)
    decoded=[tok.decode(ids[:i],skip_special_tokens=False,clean_up_tokenization_spaces=False) for i in range(len(ids)+1)]
    if decoded[-1]!=text: raise ValueError('Full canvas roundtrip failed')
    def span(lo,hi,cover=False):
        if cover:
            # BPE can merge a leading space or punctuation with the expression.
            # Include every token overlapping the character interval.
            return [i for i in range(len(ids)) if len(decoded[i])<hi and len(decoded[i+1])>lo]
        if text[:lo] not in decoded or text[:hi] not in decoded: raise ValueError('Non-token-aligned boundary')
        return list(range(decoded.index(text[:lo]),decoded.index(text[:hi])))
    ss={}
    for m in re.finditer(r'([AB]) = (00)',text[len(prefix):]):
        ss[m[1]]=span(len(prefix)+m.start(2),len(prefix)+m.end(2))
    if set(ss)!=set(requested): raise ValueError('Missing requested fields')
    sites={}
    fact_start=text.index('Facts: ')
    xy=[]
    for k in ('x','y'):
        m=re.search(rf'{k} = (\d)',text[fact_start:])
        xy+=span(fact_start+m.start(1),fact_start+m.end(1))
    sites['fact_xy']=xy
    expression=f"({p['x']} + {p['y']})" if style=='numeric' else '(x + y)'
    pos=text.index(expression,text.index('Definition of A:'))
    sites['expression_A']=span(pos,pos+len(expression),cover=True)
    for k in requested: sites['answer_'+k]=ss[k]
    sites['answer_all']=[v for k in requested for v in ss[k]]
    # Audit the full-context tokenization of every two-digit value used here.
    expected=gold(p,relation)
    for k, ps in ss.items():
        altered=text[:]
        offset=text.index(f'{k} = 00',len(prefix))+4
        altered=altered[:offset]+str(expected[k])+altered[offset+2:]
        tt=tok.encode(altered,add_special_tokens=False)
        if len(tt)!=len(ids) or any(tt[i]!=ids[i] for i in range(len(ids)) if i not in ps):
            raise ValueError('Gold numeral changes surrounding tokenization')
        for i in ps: ids[i]=b.mask_id
        # Restore placeholder for next independent audit.
        ids=tok.encode(text,add_special_tokens=False)
    for ps in ss.values():
        for i in ps: ids[i]=b.mask_id
    return dict(ids=ids,spans=ss,sites=sites,prompt=user,serialized=text,requested=requested)


def configurations():
    cc=[dict(name=f'residual-L{l+1}-{s}',kind='residual',layer=l,site=s)
        for l in (7,15,23,30) for s in ('fact_xy','expression_A','answer_A','answer_B')]
    cc.append(dict(name='bank-L25-32-odd-answer_all',kind='bank',site='answer_all',
                   bank={l:list(range(1,32,2)) for l in range(24,32)}))
    return cc


def rollout(b,target,source=None,config=None,mode='sustained'):
    if mode not in ('single','sustained'): raise ValueError(mode)
    ids=target['ids'].copy(); positions=[p for ps in target['spans'].values() for p in ps]
    if config and config['site'] not in target['sites']: raise ValueError('Absent intervention site')
    if source:
        if target['spans']!=source['spans'] or target['sites']!=source['sites'] or len(ids)!=len(source['ids']):
            raise ValueError('Source/Target token positions differ')
    writes=[]; nfe=0
    layers=b._model.model.layers if config else None
    def hooks(stack,values=None):
        site=target['sites'][config['site']]
        if config['kind']=='residual':
            return [stack.enter_context(ResidualPatch(layers[config['layer']],site,None if values is None else values[0]))]
        return [stack.enter_context(BankPatch(layers[int(l)].self_attn.o_proj,site,heads,
                None if values is None else values[j])) for j,(l,heads) in enumerate(config['bank'].items())]
    while any(ids[p]==b.mask_id for p in positions):
        missing=[p for p in positions if ids[p]==b.mask_id]
        active=config is not None and (mode=='sustained' or not writes)
        aligned=None
        if active:
            aligned=source['ids'].copy()
            for p in positions: aligned[p]=ids[p]
            with ExitStack() as stack:
                caps=hooks(stack); b.predict(aligned,missing,[[] for _ in missing])
            values=[h.captured for h in caps]; nfe+=1
            with ExitStack() as stack:
                hooks(stack,values); pred=b.predict(ids,missing,[[] for _ in missing])
        else: pred=b.predict(ids,missing,[[] for _ in missing])
        nfe+=1
        j=max(range(len(missing)),key=lambda i:(pred[i]['confidence'],-missing[i]))
        if pred[j]['token_id']==b.mask_id: raise ValueError('Mask selected')
        writes.append(dict(position=missing[j],**pred[j],patched=active,
            target_state=ids.copy(),source_state=aligned,
            predictions=[dict(position=p,**q) for p,q in zip(missing,pred)]))
        ids[missing[j]]=pred[j]['token_id']
    return dict(ids=ids,writes=writes,nfe=nfe,**read_values(b,ids,target['spans']))


def expected_cf(p,rel,kind):
    if kind in ('self','same_value','same_C_diff_answers'): return gold(p,rel)
    if kind in ('diff_C_same_answers','diff_C_diff_answers','C_only'): return gold(p,rel,c=p['x']+p['y']+1)
    return gold(sources(p,rel)[kind],rel)
