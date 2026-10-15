"""Verify and summarize the v2 calibration, retaining paired input groups."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling import precommit as v1, precommit_v2 as v2


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def paired_interval(values,seed=123):
    values=list(values);rng=random.Random(seed)
    draws=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(4000))
    return [draws[99],draws[3899]]


def counts(rows):
    return dict(n=len(rows),correct=sum(r['correct'] for r in rows),valid=sum(r['valid'] for r in rows),
                accuracy=sum(r['correct'] for r in rows)/len(rows))


def analyze(run_root,v1_root):
    data=v2.dataset()
    groups={g['id']:g for g in data['groups'] if g['split']=='calibration'}
    report=dict(seed=123,models={})
    for model in ('illada','dream'):
        run=run_root/f'{model}-v2'
        meta=json.loads((run/'manifest.json').read_text())
        assert meta['status'] in ('calibrated_v2','calibration_below_gate_v2')
        assert meta['seed']==123 and meta['dataset_sha256']==v1.fingerprint(data)
        assert meta['backend']['weights_loaded'] and 'A100' in meta['backend']['gpu_name']
        for name,digest in meta['code_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        rows=read_rows(run/'calibration.jsonl')
        expected_keys={(case,rel,req,u) for case in groups for rel in v2.RELATIONS
                       for req in ('AB','BA','A','B') for u in (False,True)}
        keys={(r['case'],r['relation'],r['requested'],r['unused']) for r in rows}
        assert len(rows)==720 and keys==expected_keys
        for r in rows:
            p=groups[r['case']]['p']
            assert r['p']==p and r['style']=='numeric' and r['ending']=='native'
            assert r['expected']=={k:v2.gold(p,r['relation'])[k] for k in r['requested']}
            assert r['canvas']['prompt']==v2.prompt(p,r['relation'],r['requested'])
            cc=r['canvas'];mask=meta['backend']['mask_id']
            assert all(cc['ids'][i]==mask for ps in cc['spans'].values() for i in ps)
            assert all(r['ids'][i]==mask for i in cc['unused'])
            assert [r['ids'][i] for i in cc['suffix_positions']]==cc['suffix_ids']
            assert r['correct']==(r['values']==r['expected'])
        assert v2.score_gate(rows)==meta['gate']
        cells=[]
        for rel in v2.RELATIONS:
            for req in ('A','B','AB','BA'):
                for unused in (False,True):
                    rr=[r for r in rows if r['relation']==rel and r['requested']==req and r['unused']==unused]
                    cells.append(dict(relation=rel,requested=req,unused=unused,**counts(rr)))
        index={(r['case'],r['relation'],r['requested'],r['unused']):r for r in rows}
        joint_single=[]
        for rel in v2.RELATIONS:
            for answer in ('A','B'):
                values=[];joint=single=0
                for case in groups:
                    deltas=[]
                    for order in ('AB','BA'):
                        for unused in (False,True):
                            j=index[case,rel,order,unused];s=index[case,rel,answer,unused]
                            jc=j['values'][answer]==j['expected'][answer];sc=s['correct']
                            joint+=jc;single+=sc;deltas.append(int(jc)-int(sc))
                    values.append(sum(deltas)/len(deltas))
                joint_single.append(dict(relation=rel,answer=answer,paired_comparisons=120,groups=30,
                    joint_correct=joint,single_correct_repeated_for_orders=single,
                    delta=sum(values)/len(values),cluster_bootstrap_95=paired_interval(values)))
        readouts=[]
        for rel in v2.RELATIONS:
            for req in ('A','B','AB','BA'):
                rr=[r for r in rows if r['relation']==rel and r['requested']==req]
                for answer in req:
                    first_correct=final_correct=0
                    for r in rr:
                        w=r['writes'][0]
                        pp=dict(zip(w['eligible'],w['predictions']))
                        initial=''.join(pp[i]['token_text'] for i in r['canvas']['spans'][answer])
                        first_correct+=initial==str(r['expected'][answer])
                        final_correct+=r['values'][answer]==r['expected'][answer]
                    readouts.append(dict(relation=rel,requested=req,answer=answer,n=len(rr),
                        t0_argmax_correct=first_correct,final_correct=final_correct,
                        interpretation='Same-state uncommitted token argmax, not a committed answer or a joint sequence likelihood'))
        report['models'][model]=dict(status=meta['status'],gate=meta['gate'],all_outputs=counts(rows),
            single_outputs=counts([r for r in rows if len(r['requested'])==1]),
            joint_outputs=counts([r for r in rows if len(r['requested'])==2]),cells=cells,
            joint_vs_single=joint_single,t0_vs_final=readouts,audit=meta['audit'],compute=meta['compute'],
            examples=[{k:r[k] for k in ('case','relation','requested','unused','expected','values','correct')}
                      |dict(prompt=r['canvas']['prompt']) for r in rows[:8]],
            results_sha256=hashlib.sha256((run/'calibration.jsonl').read_bytes()).hexdigest())
    tail=read_rows(run_root/'illada-v2/ending.jsonl')
    assert len(tail)==240
    idx={(r['case'],r['relation'],r['requested'],r['unused'],r['ending']):r for r in tail}
    assert len(idx)==240
    old=[r for r in read_rows(v1_root/'illada-v1-calibration/results.jsonl')
         if r['style']=='numeric' and r['requested'] in ('A','B')]
    assert len(old)==120
    mismatch=[]
    for r in old:
        new=idx[r['case'],r['relation'],r['requested'],r['unused'],'bare']
        assert new['canvas']['ids']==r['canvas']['ids']
        if new['ids']!=r['ids']:mismatch.append(dict(case=r['case'],relation=r['relation'],requested=r['requested'],unused=r['unused']))
    tail_cells=[dict(ending=e,unused=u,**counts([r for r in tail if r['ending']==e and r['unused']==u]))
                for e in ('bare','native') for u in (False,True)]
    detailed=[dict(relation=rel,requested=req,ending=e,unused=u,
        **counts([r for r in tail if r['relation']==rel and r['requested']==req and r['ending']==e and r['unused']==u]))
        for rel in v2.RELATIONS for req in ('A','B') for e in ('bare','native') for u in (False,True)]
    group_effects=[]
    for case in sorted({r['case'] for r in tail}):
        contributions=[]
        for rel in v2.RELATIONS:
            for req in ('A','B'):
                c=lambda u,e:int(idx[case,rel,req,u,e]['correct'])
                contributions.append((c(True,'native')-c(False,'native'))-(c(True,'bare')-c(False,'bare')))
        group_effects.append(sum(contributions)/len(contributions))
    examples=[]
    for r in tail:
        if r['ending']=='native':
            base=idx[r['case'],r['relation'],r['requested'],r['unused'],'bare']
            if r['values']!=base['values']:
                examples.append(dict(case=r['case'],relation=r['relation'],requested=r['requested'],unused=r['unused'],
                    expected=r['expected'],bare=base['values'],native=r['values'],prompt=r['canvas']['prompt']))
    report['ending_diagnostic']=dict(cells=tail_cells,detailed=detailed,groups=10,
        replay_comparisons=120,replay_token_mismatches=mismatch,
        interaction_native_minus_bare_unused_effect=sum(group_effects)/len(group_effects),
        interaction_cluster_bootstrap_95=paired_interval(group_effects),changed_examples=examples[:10])
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',type=Path)
    ap.add_argument('--v1-runs',type=Path,default=ROOT/'runs/colab-precommit-v1/artifacts/runs')
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    result=analyze(args.runs,args.v1_runs)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    for model,r in result['models'].items():print(model,r['status'],'single',r['single_outputs'],'joint',r['joint_outputs'])
    print('ENDING',result['ending_diagnostic']['cells'])
    print('REPLAY_MISMATCHES',len(result['ending_diagnostic']['replay_token_mismatches']))


if __name__=='__main__':main()
