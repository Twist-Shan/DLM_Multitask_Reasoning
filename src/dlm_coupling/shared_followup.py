"""Fixed-layer replication and branch-specific sustained interchange controls."""
import random
import re
from .shared_causal import examples, answers, canvas, substitute, ResidualPatch


def branch_answers(p, override=None):
    v=dict(C=p['x']+p['y'], U=p['u']+p['r'], V=p['v']+p['s'])
    v.update(override or {})
    return dict(**v, A=v['C']+v['U'], B=2*v['C']+v['V'])


def cases(n=10, seed=124):
    if not 1 <= n <= 50: raise ValueError('Use 1..50 cases per suite')
    rng=random.Random(seed); result=[]
    used={tuple(sorted(p.items())) for row in examples(6,123) for p in (row['base'],row['donor'],row['same'])}
    while len(result)<n:
        p=dict(x=rng.randint(4,8),y=rng.randint(6,9),u=rng.randint(2,5),v=rng.randint(2,5))
        d=dict(x=p['x']+1,y=p['y'],u=p['u']+3,v=p['v']+3)
        same=dict(x=p['x']-1 if p['y']<9 else p['x']+1,
                  y=p['y']+1 if p['y']<9 else p['y']-1,u=p['u']+3,v=p['v']+3)
        signatures={tuple(sorted(q.items())) for q in (p,d,same)}
        if signatures & used: continue
        used.update(signatures)
        result.append(dict(id=f'replication-{len(result)}',suite='replication',base=p,donor=d,same=same))
    used=set()
    while len(result)<2*n:
        p=dict(x=rng.randint(4,7),y=rng.randint(6,8),u=rng.randint(4,6),r=rng.randint(6,8),v=rng.randint(3,6),s=rng.randint(7,8))
        d=dict(p,x=p['x']+1,u=p['u']+2,v=p['v']+3)
        same=dict(p,x=p['x']-1,y=p['y']+1,u=p['u']-1,r=p['r']+1,v=p['v']-1,s=p['s']+1)
        signatures={tuple(sorted(q.items())) for q in (p,d,same)}
        if signatures & used: continue
        used.update(signatures)
        result.append(dict(id=f'branches-{len(result)-n}',suite='branches',base=p,donor=d,same=same))
    return result


def gold(case, variant='base', site=None):
    p=case[variant]
    if case['suite']=='replication': return answers(p, answers(case['donor'])['C'] if site else None)
    override={site:branch_answers(case['donor'])[site]} if site else None
    return branch_answers(p,override)


def prepare(backend,case,variant='base',visible=True):
    p=case[variant]
    if case['suite']=='replication':
        ids,spans,_,prompt=canvas(backend,p,answers(p)['C'] if visible else None)
        return ids,spans,prompt
    tok=backend.tokenizer
    prompt=(f"Compute C = {p['x']} + {p['y']}; U = {p['u']} + {p['r']}; V = {p['v']} + {p['s']}; "
            "A = C + U; B = 2 * C + V. Fill the five integer values in the supplied answer template. No explanation.\n"
            "Example: C = 4 + 6; U = 5 + 7; V = 6 + 8; A = C + U; B = 2 * C + V. "
            "Answer: C = 10; U = 12; V = 14; A = 22; B = 34.")
    prefix=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
    text=prefix+'C = 00; U = 00; V = 00; A = 00; B = 00.'
    ids=tok.encode(text,add_special_tokens=False)
    decoded=[tok.decode(ids[:i],skip_special_tokens=False,clean_up_tokenization_spaces=False) for i in range(len(ids)+1)]
    if decoded[-1]!=text: raise ValueError('Tokenizer roundtrip failed')
    spans={}
    for m in re.finditer(r'([CUVAB]) = (00)',text[len(prefix):]):
        lo,hi=len(prefix)+m.start(2),len(prefix)+m.end(2)
        spans[m[1]]=list(range(decoded.index(text[:lo]),decoded.index(text[:hi])))
    if set(spans)!=set('CUVAB'): raise ValueError('Incomplete canvas')
    for positions in spans.values():
        for pos in positions: ids[pos]=backend.mask_id
    if visible:
        for key in 'CUV': substitute(backend,ids,spans[key],branch_answers(p)[key])
    return ids,spans,prompt


def aligned_donor(donor,recipient,answer_positions,mask_id):
    if len(donor)!=len(recipient): raise ValueError('Unequal canvas lengths')
    out=donor.copy()
    for pos in answer_positions: out[pos]=recipient[pos]
    if [x==mask_id for x in out]!=[x==mask_id for x in recipient]:
        raise ValueError('Different source/recipient mask patterns')
    return out


def rollout(backend,state,spans,module=None,donor=None,site=None,mode='single'):
    """Recompute donor from exactly recipient's partial A/B on each patched step."""
    if mode not in ('single','sustained'): raise ValueError('Unknown patch mode')
    ids=state.copy(); positions=spans['A']+spans['B']; writes=[]; nfe=0
    while any(ids[p]==backend.mask_id for p in positions):
        missing=[p for p in positions if ids[p]==backend.mask_id]
        active=module is not None and (mode=='sustained' or not writes)
        aligned=None
        if active:
            aligned=aligned_donor(donor,ids,positions,backend.mask_id)
            with ResidualPatch(module,site) as cap:
                backend.predict(aligned,missing,[[] for _ in missing])
            nfe+=1
            with ResidualPatch(module,site,cap.captured):
                preds=backend.predict(ids,missing,[[] for _ in missing])
        else: preds=backend.predict(ids,missing,[[] for _ in missing])
        nfe+=1
        j=max(range(len(missing)),key=lambda i:(preds[i]['confidence'],-missing[i]))
        if preds[j]['token_id']==backend.mask_id: raise ValueError('Mask selected')
        writes.append(dict(position=missing[j],token_id=preds[j]['token_id'],patched=active,
                           donor_state=aligned,recipient_state=ids.copy(),
                           predictions=[dict(position=p,**v) for p,v in zip(missing,preds)]))
        ids[missing[j]]=preds[j]['token_id']
    return ids,writes,nfe
