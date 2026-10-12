import json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling.backends.hf import HFBackend
from dlm_coupling.baseline_repair import make_canvas
from dlm_coupling.shared_causal import fill,read_values
from dlm_coupling.shared_followup import branch_answers
from dlm_coupling.mechanism_pilot import dataset,rollout
from dlm_coupling.headbank_pilot import bank_rollout,validation_cases

def main():
    out=ROOT/'runs/headbank-pilot';out.mkdir(parents=True,exist_ok=False)
    oldroot=ROOT/'runs/mechanism-pilot';oldrows=[json.loads(s) for s in (oldroot/'results.jsonl').read_text().splitlines()]
    discovery=[c for c in dataset() if c['split']=='discovery'];validation=validation_cases()
    manifest=dict(status='running',data=discovery+validation,discovery_seed=127,validation_seed=129,
        selection='Layer bank: max discovery eligible induced; min head count; lexical tie. Head subset: same rule within selected layers. Fixed before seed129 inference.',
        limitation='Exploratory multi-stage discovery; all32layers/all32heads AB patch is an algebraic positive control, not circuit discovery.')
    def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save();b=HFBackend(dict(family='illada',model_id='GSAI-ML/iLLaDA-8B-Instruct',revision='5769f049224b2cff61106ebce0429ede5deeb434',trust_remote_code=True,dtype='bfloat16',device='cuda'),load_model=True)
    b._torch.manual_seed(127);rows=[];states={};nfe=0
    banks={name:{l:list(range(32)) for l in ll} for name,ll in [('all_layers',range(32)),('layers1_8',range(8)),('layers9_16',range(8,16)),('layers17_24',range(16,24)),('layers25_32',range(24,32)),('layers1_16',range(16)),('layers17_32',range(16,32))]}
    with (out/'results.jsonl').open('w',encoding='utf-8') as f:
        def emit(r):rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
        for c in discovery:
            st={}
            for v in c['variants']:
                r=next(r for r in oldrows if r['case']==c['id'] and r['kind']=='baseline' and r['variant']==v)
                mid=r['initial_state'].copy()
                for w in r['writes']:
                    if w['position'] in [p for k in 'CUV' for p in r['spans'][k]]:mid[w['position']]=w['token_id']
                st[v]=dict(mid=mid,end=r['ids'],values=r['values'],all_correct=r['values']==branch_answers(c['variants'][v]))
            states[c['id']]=(st,r['spans'])
        def trial(c,name,bank,target='C',control=None,phase='layer_discovery'):
            nonlocal nfe
            st,ss=states[c['id']];v=control or target
            end,w,n=bank_rollout(b,st['base']['mid'],st[v]['mid'],ss,bank);nfe+=n
            vals=read_values(b,end,ss)['values'];expect=branch_answers(c['variants']['base' if control else target]);baseline=st['base']['values']
            correct=all(vals[k]==expect[k] for k in 'AB')
            r=dict(case=c['id'],split=c['split'],phase=phase,name=name,bank=bank,size=sum(map(len,bank.values())),target=target,control=control,
                   initial_state=st['base']['mid'],spans=ss,ids=end,writes=w,nfe=n,values=vals,expected=expect,baseline_values=baseline,
                   eligible=st['base']['all_correct'] and st[v]['all_correct'],correct=correct,induced=correct and any(baseline[k]!=expect[k] for k in 'AB'))
            if control=='base':assert end==st['base']['end']
            if name=='all_layers' and control is None:assert all(vals[k]==st[v]['values'][k] for k in 'AB'),'All-layer attention replacement failed donor equivalence'
            emit(r);return r
        def pick(names,phase):
            def score(name):
                rr=[r for r in rows if r['phase']==phase and r['name']==name]
                return (sum(r['induced'] and r['eligible'] for r in rr),sum(r['correct'] for r in rr),-rr[0]['size'],name)
            return max(sorted(names),key=score)
        for c in discovery:
            for name,bank in banks.items():trial(c,name,bank)
            trial(c,'all_layers',banks['all_layers'],control='base',phase='self_check')
            print('LAYER BANKS',c['id'],flush=True)
        layer_choice=pick(banks,'layer_discovery');selected_layers=list(banks[layer_choice]);manifest['layer_choice']=layer_choice
        # Group candidates span all head indices; no individual-head effect presumed.
        headsets={f'quarter{i}':list(range(i*8,(i+1)*8)) for i in range(4)}
        headsets.update(even=list(range(0,32,2)),odd=list(range(1,32,2)),all=list(range(32)))
        rng=random.Random(127)
        for i in range(3):headsets[f'random8_{i}']=sorted(rng.sample(range(32),8))
        subsets={name:{l:hh for l in selected_layers} for name,hh in headsets.items()}
        for c in discovery:
            for name,bank in subsets.items():trial(c,name,bank,phase='head_subset_discovery')
            print('HEAD SUBSETS',c['id'],flush=True)
        subset_choice=pick(subsets,'head_subset_discovery');selected=subsets[subset_choice]
        selected_heads=headsets[subset_choice];rr=random.Random(129)
        controls={f'matched_random{i}':{l:sorted(rr.sample(range(32),len(hh))) for l,hh in selected.items()} for i in range(3)}
        complement={l:[h for h in range(32) if h not in hh] for l,hh in selected.items()}
        complement={l:hh for l,hh in complement.items() if hh}
        manifest.update(subset_choice=subset_choice,selected=selected,controls=controls,complement=complement,layer_candidates=banks,subset_candidates=subsets);save()
        print('FROZEN BANK',layer_choice,subset_choice,'size',sum(map(len,selected.values())),flush=True)
        b._torch.manual_seed(129)
        for c in validation:
            st={}
            for v,p in c['variants'].items():
                initial,ss,prompt=make_canvas(b,dict(suite='branches',base=p),'generated_intermediates')
                mid,w=fill(b,initial,[pos for k in 'CUV' for pos in ss[k]]);end,w2,n=rollout(b,mid,ss);nfe+=len(w)+n
                vals=read_values(b,end,ss)['values'];expect=branch_answers(p)
                st[v]=dict(mid=mid,end=end,values=vals,all_correct=vals==expect)
                emit(dict(case=c['id'],split=c['split'],phase='baseline',variant=v,initial_state=initial,spans=ss,ids=end,writes=w+w2,nfe=len(w)+n,values=vals,expected=expect,correct=all(vals[k]==expect[k] for k in 'AB'),prompt=prompt))
            states[c['id']]=(st,ss)
            for target in 'CUV':trial(c,'selected',selected,target=target,phase='validation')
            for control in ('base','same'):trial(c,'selected',selected,control=control,phase='validation')
            for name,bank in controls.items():trial(c,name,bank,phase='validation')
            trial(c,'complement',complement,phase='validation')
            trial(c,'all_layers',banks['all_layers'],phase='validation')
            print('BANK VALIDATION',c['id'],{v:s['all_correct'] for v,s in st.items()},flush=True)
    manifest.update(status='complete',rows=len(rows),nfe=nfe,backend=b.metadata);save();print('HEADBANK COMPLETE',len(rows),nfe,flush=True)
if __name__=='__main__':main()
