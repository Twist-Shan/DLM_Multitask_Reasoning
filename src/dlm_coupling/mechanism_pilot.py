"""Small discovery/validation study of generated intermediate use."""
import random
from .shared_causal import examples, ResidualPatch
from .shared_followup import cases, aligned_donor, branch_answers
from .baseline_repair import splits

LAYERS=(7,15,23,30)

def dataset():
    used={tuple(sorted(p.items())) for c in examples(6,123) for p in (c['base'],c['donor'],c['same'])}
    used.update(tuple(sorted(c[v].items())) for c in cases(10,124) for v in ('base','donor','same'))
    used.update(tuple(sorted(c['base'].items())) for rows in splits().values() for c in rows)
    result=[]
    for split,seed,n in [('discovery',127,4),('validation',128,6)]:
        rng=random.Random(seed);count=0
        for attempt in range(100000):
            if count==n: break
            p=dict(x=rng.randint(4,7),y=rng.randint(6,8),u=rng.randint(4,6),r=rng.randint(6,8),v=rng.randint(3,6),s=rng.randint(7,8))
            variants=dict(base=p,C=dict(p,x=p['x']+1),U=dict(p,u=p['u']+1),V=dict(p,v=p['v']+1),same=dict(p,x=p['x']-1,y=p['y']+1))
            keys={tuple(sorted(q.items())) for q in variants.values()}
            if len(keys)!=5 or used&keys: continue
            used.update(keys);result.append(dict(id=f'{split}-{count}',split=split,suite='branches',variants=variants));count+=1
        if count!=n: raise ValueError('Could not sample disjoint cases')
    return result


class HeadPatch:
    """Patch one query-head's pre-o_proj output; compatible with GQA."""
    def __init__(self,module,positions,head,nheads,donor=None):
        self.module,self.positions,self.head,self.nheads,self.donor=module,positions,head,nheads,donor
        self.calls=0;self.captured=None
    def __enter__(self):
        def hook(module,args):
            self.calls+=1;x=args[0]
            if x.ndim!=3 or x.shape[0]!=1 or x.shape[-1]%self.nheads: raise ValueError('Unexpected head output')
            width=x.shape[-1]//self.nheads
            if not 0<=self.head<self.nheads: raise ValueError('Bad head')
            sl=slice(self.head*width,(self.head+1)*width)
            self.captured=x[:,self.positions,sl].detach().clone()
            if self.donor is None:return None
            if self.donor.shape!=self.captured.shape: raise ValueError('Shape mismatch')
            out=x.clone();out[:,self.positions,sl]=self.donor.to(x)
            return (out,*args[1:])
        self.handle=self.module.register_forward_pre_hook(hook);return self
    def __exit__(self,*args):
        self.handle.remove()
        if args[0] is None and self.calls!=1: raise AssertionError('Expected one head hook')


def rollout(b,state,spans,module=None,donor=None,site=None,mode='sustained',head=None):
    if mode not in ('single','sustained'): raise ValueError(mode)
    ids=state.copy();positions=spans['A']+spans['B'];writes=[];nfe=0
    def hook(source=None):
        return ResidualPatch(module,site,source) if head is None else HeadPatch(module,site,head,b.model_config.num_attention_heads,source)
    while any(ids[p]==b.mask_id for p in positions):
        missing=[p for p in positions if ids[p]==b.mask_id]
        active=module is not None and (mode=='sustained' or not writes)
        aligned=None
        if active:
            aligned=aligned_donor(donor,ids,positions,b.mask_id)
            with hook() as cap:b.predict(aligned,missing,[[] for _ in missing])
            nfe+=1
            with hook(cap.captured):pred=b.predict(ids,missing,[[] for _ in missing])
        else:pred=b.predict(ids,missing,[[] for _ in missing])
        nfe+=1;j=max(range(len(missing)),key=lambda i:(pred[i]['confidence'],-missing[i]))
        if pred[j]['token_id']==b.mask_id:raise ValueError('Mask selected')
        writes.append(dict(position=missing[j],**pred[j],patched=active,recipient_state=ids.copy(),donor_state=aligned,
                           predictions=[dict(position=p,**v) for p,v in zip(missing,pred)]))
        ids[missing[j]]=pred[j]['token_id']
    return ids,writes,nfe


def choose(rows,component=None):
    candidates={r['config'] for r in rows if r['kind']=='grid' and (component is None or r['component']==component)}
    if not candidates:raise ValueError('No grid')
    def score(c):
        rr=[r for r in rows if r['kind']=='grid' and r['config']==c]
        return (sum(r['induced'] and r['eligible'] for r in rr),sum(r['correct'] for r in rr),-rr[0]['layer'],c)
    return max(sorted(candidates),key=score)
