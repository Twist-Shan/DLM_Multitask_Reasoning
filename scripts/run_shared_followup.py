"""Seed-124 follow-up: fixed layer 8; single vs dynamically aligned sustained patch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling.backends.hf import HFBackend
from dlm_coupling.shared_causal import fill, substitute, read_values
from dlm_coupling.shared_followup import cases, gold, prepare, rollout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=124);ap.add_argument('--n',type=int,default=10)
    a=ap.parse_args();dataset=cases(a.n,a.seed)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',args=vars(a),layer_zero_based=7,cases=dataset,
        selection='Layer 8 fixed before seed-124 data inference; 10 new replication and 10 new branch-control cases; all included',
        source_sha256=hashlib.sha256(b''.join(p.read_bytes() for p in [Path(__file__),ROOT/'src/dlm_coupling/shared_followup.py',ROOT/'src/dlm_coupling/shared_causal.py'])).hexdigest())
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save()
    b=HFBackend(dict(family='illada',model_id='GSAI-ML/iLLaDA-8B-Instruct',revision='5769f049224b2cff61106ebce0429ede5deeb434',trust_remote_code=True,dtype='bfloat16',device='cuda'),load_model=True)
    b._torch.manual_seed(a.seed);b._torch.cuda.manual_seed_all(a.seed)
    module=dict(b._model.named_modules())['model.layers'][7]
    nfe=0;count=0
    with (out/'results.jsonl').open('w',encoding='utf-8') as f:
        def emit(case,condition,state,spans,end,writes,expected,**extra):
            nonlocal count
            r=dict(case=case['id'],suite=case['suite'],condition=condition,initial_state=state,
                spans=spans,ids=end,writes=writes,result=read_values(b,end,spans),expected=expected,**extra)
            f.write(json.dumps(r)+'\n');f.flush();count+=1
            print(case['id'],condition,extra.get('stage',''),extra.get('mode',''),r['result']['values'],flush=True)
        for case in dataset:
            intermediates=['C'] if case['suite']=='replication' else ['C','U','V']
            for variant in ('base','donor','same'):
                state,ss,prompt=prepare(b,case,variant,False)
                gen,w=fill(b,state,[p for k in intermediates for p in ss[k]])
                end,w2=fill(b,gen,ss['A']+ss['B']);nfe+=len(w)+len(w2)
                emit(case,'generated_'+variant,state,ss,end,w+w2,gold(case,variant),prompt=prompt)
            for target in intermediates:
                state,ss,_=prepare(b,case)
                substitute(b,state,ss[target],gold(case,'donor')[target])
                end,w,n=rollout(b,state,ss);nfe+=n
                emit(case,'token_'+target,state,ss,end,w,gold(case,site=target),stage='visible')
            for visible in (True,False):
                stage='visible' if visible else 'masked'
                state,ss,prompt=prepare(b,case,visible=visible)
                control,w,n=rollout(b,state,ss);nfe+=n
                emit(case,'baseline',state,ss,control,w,gold(case),stage=stage,prompt=prompt)
                conditions=[('self','base','C'),('same','same','C'),('syntax','donor','syntax')]
                conditions += [('patch_'+k,'donor',k) for k in intermediates]
                for mode in ('single','sustained'):
                    for condition,variant,target in conditions:
                        donor,ds,_=prepare(b,case,variant,visible)
                        if ds!=ss or len(donor)!=len(state): raise ValueError('Canvas positions differ')
                        site=ss[target] if target!='syntax' else list(range(ss['C'][0]-len(ss['C']),ss['C'][0]))
                        end,w,n=rollout(b,state,ss,module,donor,site,mode);nfe+=n
                        if condition=='self' and end!=control: raise AssertionError('Self patch failed')
                        expected=gold(case,site=target) if condition.startswith('patch_') else gold(case)
                        emit(case,condition,state,ss,end,w,expected,stage=stage,mode=mode,site=site,
                             donor_gold=gold(case,variant),control_exact=end==control)
    manifest.update(status='complete',rows=count,nfe=nfe,backend=b.metadata);save()
    print('FOLLOWUP COMPLETE',count,'rows',nfe,'forwards',flush=True)


if __name__=='__main__': main()
