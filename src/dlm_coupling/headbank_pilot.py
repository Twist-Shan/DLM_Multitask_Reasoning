"""Multi-layer query-head output interchange with heldout validation."""
import random
from contextlib import ExitStack
from .mechanism_pilot import dataset
from .baseline_repair import splits
from .shared_causal import examples
from .shared_followup import cases,aligned_donor

class BankPatch:
    def __init__(self,module,positions,heads,donor=None):
        self.module,self.positions,self.heads,self.donor=module,positions,heads,donor;self.calls=0;self.captured=None
    def __enter__(self):
        def hook(module,args):
            x=args[0];self.calls+=1
            assert x.ndim==3 and x.shape[0]==1 and x.shape[-1]==4096
            assert len(self.heads)==len(set(self.heads)) and all(0<=h<32 for h in self.heads)
            channels=[j for h in self.heads for j in range(h*128,(h+1)*128)]
            self.captured=x[:,self.positions,:][...,channels].detach().clone()
            if self.donor is None:return None
            assert self.donor.shape==self.captured.shape
            out=x.clone();selected=out[:,self.positions,:].clone();selected[...,channels]=self.donor.to(x);out[:,self.positions,:]=selected
            return (out,*args[1:])
        self.handle=self.module.register_forward_pre_hook(hook);return self
    def __exit__(self,*args):
        self.handle.remove()
        if args[0] is None:assert self.calls==1

def bank_rollout(b,base,donor,spans,bank):
    ids=base.copy();positions=spans['A']+spans['B'];writes=[]
    layers=b._model.model.layers
    while any(ids[p]==b.mask_id for p in positions):
        missing=[p for p in positions if ids[p]==b.mask_id];aligned=aligned_donor(donor,ids,positions,b.mask_id)
        with ExitStack() as stack:
            caps={l:stack.enter_context(BankPatch(layers[l].self_attn.o_proj,positions,hh)) for l,hh in bank.items()}
            b.predict(aligned,missing,[[] for _ in missing])
        with ExitStack() as stack:
            for l,hh in bank.items():stack.enter_context(BankPatch(layers[l].self_attn.o_proj,positions,hh,caps[l].captured))
            pp=b.predict(ids,missing,[[] for _ in missing])
        j=max(range(len(missing)),key=lambda i:(pp[i]['confidence'],-missing[i]));assert pp[j]['token_id']!=b.mask_id
        writes.append(dict(position=missing[j],**pp[j],recipient_state=ids.copy(),donor_state=aligned,predictions=[dict(position=p,**v) for p,v in zip(missing,pp)]))
        ids[missing[j]]=pp[j]['token_id']
    return ids,writes,len(writes)*2

def validation_cases():
    old={tuple(sorted(p.items())) for c in examples(6,123) for p in (c['base'],c['donor'],c['same'])}
    old.update(tuple(sorted(c[v].items())) for c in cases(10,124) for v in ('base','donor','same'))
    old.update(tuple(sorted(c['base'].items())) for rr in splits().values() for c in rr)
    old.update(tuple(sorted(p.items())) for c in dataset() for p in c['variants'].values())
    rng=random.Random(129);result=[]
    for _ in range(100000):
        if len(result)==6:return result
        p=dict(x=rng.randint(4,7),y=rng.randint(6,8),u=rng.randint(4,6),r=rng.randint(6,8),v=rng.randint(3,6),s=rng.randint(7,8))
        vv=dict(base=p,C=dict(p,x=p['x']+1),U=dict(p,u=p['u']+1),V=dict(p,v=p['v']+1),same=dict(p,x=p['x']-1,y=p['y']+1))
        keys={tuple(sorted(q.items())) for q in vv.values()}
        if len(keys)!=5 or keys&old:continue
        old.update(keys);result.append(dict(id=f'bank-validation-{len(result)}',split='bank-validation',suite='branches',variants=vv))
    raise ValueError('No disjoint cases')
