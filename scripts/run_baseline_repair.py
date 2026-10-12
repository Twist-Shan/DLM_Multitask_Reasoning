import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling.baseline_repair import splits,VARIANTS,run_one,choose
from dlm_coupling.shared_followup import gold
from dlm_coupling.backends.hf import HFBackend

def main():
    out=ROOT/'runs/baseline-repair';out.mkdir(parents=True,exist_ok=False)
    data=splits();manifest=dict(status='running',data=data,calibration_seed=125,heldout_seed=126,parser_version=2,
        variants=VARIANTS,selection='Most calibration pairs correct, then fewest total forwards, then declared variant order; old control excluded',
        source_sha256=hashlib.sha256((ROOT/'src/dlm_coupling/baseline_repair.py').read_bytes()).hexdigest())
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save()
    b=HFBackend(dict(family='illada',model_id='GSAI-ML/iLLaDA-8B-Instruct',revision='5769f049224b2cff61106ebce0429ede5deeb434',trust_remote_code=True,dtype='bfloat16',device='cuda'),load_model=True)
    rows=[]
    with (out/'results.jsonl').open('w',encoding='utf-8') as f:
        def run(case,variant,split):
            r=dict(case=case['id'],suite=case['suite'],split=split,variant=variant,expected=gold(case),**run_one(b,case,variant))
            r['correct']=all(r['values'].get(k)==r['expected'][k] for k in 'AB')
            rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
            print(r['case'],variant,r['values'],'correct',r['correct'],'nfe',r['nfe'],flush=True)
        b._torch.manual_seed(125);b._torch.cuda.manual_seed_all(125)
        for c in data['calibration']:
            for v in VARIANTS: run(c,v,'calibration')
        chosen={s:choose(rows,s) for s in ('replication','branches')}
        manifest['chosen']=chosen;save();print('FROZEN SELECTION',chosen,flush=True)
        b._torch.manual_seed(126);b._torch.cuda.manual_seed_all(126)
        for c in data['heldout']:
            for v in ('old_masked',chosen[c['suite']]): run(c,v,'heldout')
    summary=[]
    for split in data:
        for suite in ('replication','branches'):
            for v in VARIANTS:
                group=[r for r in rows if (r['split'],r['suite'],r['variant'])==(split,suite,v)]
                if group: summary.append(dict(split=split,suite=suite,variant=v,n=len(group),correct=sum(r['correct'] for r in group),nfe=sum(r['nfe'] for r in group)))
    manifest.update(status='complete',backend=b.metadata,summary=summary,rows=len(rows),nfe=sum(r['nfe'] for r in rows));save()
    print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__': main()
