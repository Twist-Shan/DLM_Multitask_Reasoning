import hashlib,inspect,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling.backends.hf import HFBackend
from dlm_coupling.baseline_repair import make_canvas
from dlm_coupling.shared_causal import fill,read_values,ResidualPatch
from dlm_coupling.shared_followup import branch_answers
from dlm_coupling.mechanism_pilot import dataset,LAYERS,rollout,choose

def main():
    import numpy as np
    out=ROOT/'runs/mechanism-pilot';out.mkdir(parents=True,exist_ok=False)
    data=dataset();manifest=dict(status='running',data=data,layers_zero_based=LAYERS,seeds=[127,128],
        selection='Discovery only: eligible induced successes, all-case counterfactual successes, earlier layer, config lexical tie; head: eligible induced, all correct, lower head index',
        source_sha256=hashlib.sha256((ROOT/'src/dlm_coupling/mechanism_pilot.py').read_bytes()).hexdigest())
    def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save();b=HFBackend(dict(family='illada',model_id='GSAI-ML/iLLaDA-8B-Instruct',revision='5769f049224b2cff61106ebce0429ede5deeb434',trust_remote_code=True,dtype='bfloat16',device='cuda'),load_model=True)
    layers=dict(b._model.named_modules())['model.layers'];print('BLOCK',layers[7],flush=True)
    # Save exact architecture code used for hooks, including head layout.
    for obj in (type(b._model),type(layers[7]),type(layers[7].self_attn)):
        p=Path(inspect.getfile(obj));(out/p.name).write_bytes(p.read_bytes())
    manifest['modules']={str(l):[n for n,_ in layers[l].named_modules()] for l in LAYERS}
    for l in LAYERS:
        assert all(hasattr(layers[l],k) for k in ('self_attn','mlp'))
        assert hasattr(layers[l].self_attn,'o_proj')
    save();rows=[];states={};arrays={};nfe=0
    configs={f'{comp}-L{l+1}-{loc}':dict(component=comp,layer=l,location=loc) for comp in ('residual','attention','mlp') for l in LAYERS for loc in ('C','AB')}
    def module(cfg,head=None):
        block=layers[cfg['layer']]
        if head is not None:return block.self_attn.o_proj
        return block if cfg['component']=='residual' else block.self_attn if cfg['component']=='attention' else block.mlp
    with (out/'results.jsonl').open('w',encoding='utf-8') as f:
        def emit(r):rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
        def setup(c):
            nonlocal nfe
            state={};spans=None
            for v,p in c['variants'].items():
                initial,ss,prompt=make_canvas(b,dict(suite='branches',base=p),'generated_intermediates')
                if spans is not None:assert ss==spans
                spans=ss;mid,w=fill(b,initial,[pos for k in 'CUV' for pos in ss[k]])
                end,w2,n=rollout(b,mid,ss);nfe+=len(w)+n
                val=read_values(b,end,ss)['values'];expected=branch_answers(p)
                state[v]=dict(initial=initial,mid=mid,end=end,values=val,all_correct=val==expected)
                emit(dict(case=c['id'],split=c['split'],kind='baseline',variant=v,prompt=prompt,initial_state=initial,spans=ss,ids=end,writes=w+w2,nfe=len(w)+n,values=val,expected=expected,correct=all(val[k]==expected[k] for k in 'AB')))
            # Pre/post numeric positions; all four layers captured in one forward.
            positions=[p for k in 'CUVAB' for p in spans[k]]
            from contextlib import ExitStack
            for v in ('base','C','same'):
                for stage in ('initial','mid'):
                    with ExitStack() as stack:
                        caps=[stack.enter_context(ResidualPatch(layers[l],positions)) for l in LAYERS]
                        b.predict(state[v][stage],spans['A']+spans['B'],[[]]*4);nfe+=1
                    arrays[c['id']+'_'+v+'_'+stage]=np.stack([x.captured.float().cpu().numpy().astype(np.float16)[0] for x in caps])
            np.savez_compressed(out/'representations.npz',**arrays)
            states[c['id']]=(state,spans)
            print('BASELINES',c['id'],{v:s['all_correct'] for v,s in state.items()},flush=True)
        def trial(c,name,cfg,target='C',control=None,mode='sustained',head=None,kind='grid',stage='mid'):
            nonlocal nfe
            st,ss=states[c['id']];variant=control if control in ('base','same') else target
            base=st['base'][stage];donor=st[variant][stage]
            loc=cfg['location'];site=ss[target] if loc=='C' else ss['A']+ss['B']
            if control=='syntax':site=list(range(ss['C'][0]-len(ss['C']),ss['C'][0]))
            end,w,n=rollout(b,base,ss,module(cfg,head),donor,site,mode,head);nfe+=n
            val=read_values(b,end,ss)['values'];expected=branch_answers(c['variants']['base' if control else target])
            baseline=st['base']['values']
            if stage=='initial':
                ref,_,_=rollout(b,base,ss);nfe+=4;baseline=read_values(b,ref,ss)['values']
            correct=all(val[k]==expected[k] for k in 'AB')
            r=dict(case=c['id'],split=c['split'],kind=kind,config=name,**cfg,target=target,control=control,mode=mode,head=head,stage=stage,
                   initial_state=base,spans=ss,site=site,ids=end,writes=w,nfe=n+(4 if stage=='initial' else 0),values=val,expected=expected,
                   baseline_values=baseline,eligible=st['base']['all_correct'] and st[variant]['all_correct'],correct=correct,
                   induced=correct and any(baseline[k]!=expected[k] for k in 'AB'))
            if control=='base':assert end==st['base']['end'] if stage=='mid' else all(val[k]==baseline[k] for k in 'AB')
            emit(r);return r
        for split in ('discovery','validation'):
            b._torch.manual_seed(127 if split=='discovery' else 128)
            for c in [c for c in data if c['split']==split]:
                setup(c)
                if split=='discovery':
                    for name,cfg in configs.items():trial(c,name,cfg)
                    print('GRID COMPLETE',c['id'],flush=True)
                else:
                    for name in dict.fromkeys((manifest['chosen'],manifest['attention_chosen'])):
                        cfg=configs[name]
                        for target in 'CUV':trial(c,name,cfg,target=target,kind='validation')
                        for control in ('base','same','syntax'):trial(c,name,cfg,control=control,kind='validation')
                        trial(c,name,cfg,mode='single',kind='validation')
                        trial(c,name,cfg,kind='validation',stage='initial')
                    if manifest['head_chosen'] is not None:
                        cfg=dict(configs[manifest['attention_chosen']],location='AB')
                        for head in (manifest['head_chosen'],(manifest['head_chosen']+16)%32):
                            for target in 'CUV':trial(c,'head',cfg,target=target,head=head,kind='head_validation')
                            for control in ('base','same'):trial(c,'head',cfg,control=control,head=head,kind='head_validation')
                    print('VALIDATION COMPLETE',c['id'],flush=True)
            if split=='discovery':
                manifest['chosen']=choose(rows);manifest['attention_chosen']=choose([r for r in rows if r.get('location')=='AB'],'attention')
                cfg=configs[manifest['attention_chosen']];headrows=[]
                # Head scan is exploratory even if whole-attention intervention is weak.
                for c in [c for c in data if c['split']=='discovery']:
                    for head in range(32):headrows.append(trial(c,'head',cfg,head=head,kind='head_discovery'))
                    print('HEAD SCAN COMPLETE',c['id'],flush=True)
                def score(h):
                    rr=[r for r in headrows if r['head']==h]
                    return (sum(r['induced'] and r['eligible'] for r in rr),sum(r['correct'] for r in rr),-h)
                best=max(range(32),key=score)
                manifest['head_chosen']=best if score(best)[0]>0 else None
                save();print('FROZEN SELECTION',manifest['chosen'],manifest['attention_chosen'],manifest['head_chosen'],flush=True)
    manifest.update(status='complete',rows=len(rows),nfe=nfe,representation_forwards=len(data)*6,backend=b.metadata);save()
    print('MECHANISM COMPLETE',len(rows),'rows',nfe,'forwards',flush=True)
if __name__=='__main__':main()
