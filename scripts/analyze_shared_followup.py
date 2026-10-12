"""Independent arithmetic and per-step intervention audit; paired summary."""
import argparse
from collections import defaultdict
import hashlib,json,math,random
from pathlib import Path


def oracle(case, variant='base', target=None):
    p=case[variant]; c=p['x']+p['y']
    if case['suite']=='replication':
        if target: c=case['donor']['x']+case['donor']['y']
        return dict(C=c,A=c+p['u'],B=2*c+p['v'])
    values=dict(C=c,U=p['u']+p['r'],V=p['v']+p['s'])
    if target:
        q=case['donor']; pair={'C':('x','y'),'U':('u','r'),'V':('v','s')}[target]
        values[target]=q[pair[0]]+q[pair[1]]
    return dict(**values,A=values['C']+values['U'],B=2*values['C']+values['V'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();root=Path(a.run)
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert m['status']=='complete' and m['layer_zero_based']==7
    rows=[json.loads(s) for s in (root/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    cases={c['id']:c for c in m['cases']}
    assert len(rows)==m['rows']==54*m['args']['n']
    assert len({(r['case'],r['condition'],r.get('stage'),r.get('mode')) for r in rows})==len(rows)
    groups=defaultdict(list); nfe=0;selfs=0
    baselines={(r['case'],r['stage']):r for r in rows if r['condition']=='baseline'}
    eligible={c:True for c in cases}
    for r in rows:
        condition=r['condition'];case=cases[r['case']]
        target=condition.split('_')[1] if condition.startswith(('patch_','token_')) else None
        variant=condition[10:] if condition.startswith('generated_') else 'base'
        expected=oracle(case,variant,target)
        assert r['expected']==expected
        if condition in ('generated_base','generated_donor'):
            eligible[r['case']] &= all(r['result']['values'][k]==v for k,v in expected.items())
        current=r['initial_state'].copy();ss=r['spans']; answer_positions=ss['A']+ss['B'];committed_text={}
        answer_masks=[p for p in answer_positions if current[p]==5]
        assert answer_masks==answer_positions
        writes=r['writes'];nfe+=len(writes)
        for step,w in enumerate(writes):
            pos=w['position'];assert current[pos]==5 and w['token_id']!=5
            selected=next(pred for pred in w['predictions'] if pred['position']==pos)
            assert selected['token_id']==w['token_id']
            committed_text[pos]=selected['token_text']
            assert pos in [p for pp in ss.values() for p in pp]
            if 'recipient_state' in w:
                assert w['recipient_state']==current
                should_patch='mode' in r and (r['mode']=='sustained' or step==0)
                assert w['patched']==should_patch
                missing=[p for p in answer_positions if current[p]==5]
                assert [v['position'] for v in w['predictions']]==missing
                chosen=max(w['predictions'],key=lambda v:(v['confidence'],-v['position']))
                assert chosen['position']==pos and chosen['token_id']==w['token_id']
                if should_patch:
                    d=w['donor_state'];nfe+=1
                    assert len(d)==len(current)
                    assert [v==5 for v in d]==[v==5 for v in current]
                    assert all(d[p]==current[p] for p in answer_positions)
                    assert not set(r['site']) & set(answer_positions)
                    if condition=='self': assert d==current
                else: assert w['donor_state'] is None
            current[pos]=w['token_id']
        assert current==r['ids']
        for k in ('A','B'):
            text=''.join(committed_text[p] for p in ss[k])
            assert text==r['result']['texts'][k]
            value=int(text) if len(text)==2 and text.isdecimal() else None
            assert value==r['result']['values'][k]
        if condition=='self':
            assert r['ids']==baselines[(r['case'],r['stage'])]['ids'];selfs+=1
        r['both']=all(r['result']['values'][k]==expected[k] for k in ('A','B'))
        if condition.startswith(('patch_','token_')):
            baseline=baselines[(r['case'],r['stage'])]['result']['values']
            r['preexisting_target']=all(baseline[k]==expected[k] for k in ('A','B'))
            r['induced_both']=r['both'] and not r['preexisting_target']
        if not condition.startswith('generated_'):
            preds={p['position']:p['token_text'] for p in writes[0]['predictions']}
            immediate={k:''.join(preds[p] for p in ss[k]) for k in ('A','B')}
            r['immediate_both']=all(immediate[k]==str(expected[k]) for k in ('A','B'))
        else:r['immediate_both']=None
        groups[(r['suite'],condition,r.get('stage',''),r.get('mode',''))].append(r)
    assert nfe==m['nfe']
    summary=[]
    for (suite,condition,stage,mode),rs in groups.items():
        qualified=[r for r in rs if eligible[r['case']]]
        row=dict(suite=suite,condition=condition,stage=stage,mode=mode,n=len(rs),
                 final_both=sum(r['both'] for r in rs),valid_AB=sum(all(r['result']['values'][k] is not None for k in ('A','B')) for r in rs),
                 immediate_both=None if rs[0]['immediate_both'] is None else sum(r['immediate_both'] for r in rs),
                 baseline_and_donor_correct_n=len(qualified),qualified_final_both=sum(r['both'] for r in qualified))
        if condition.startswith(('patch_','token_')):
            row.update(induced_counterfactual_both=sum(r['induced_both'] for r in rs),
                       preexisting_target_matches=sum(r['preexisting_target'] for r in rs))
        if condition in ('patch_U','patch_V'):
            affected,other=('A','B') if condition=='patch_U' else ('B','A')
            row.update(affected_correct=sum(r['result']['values'][affected]==r['expected'][affected] for r in rs),
                       other_unchanged=sum(r['result']['values'][other]==baselines[(r['case'],stage)]['result']['values'][other] for r in rs))
        summary.append(row)
    contrasts=[];rng=random.Random(m['args']['seed'])
    for (suite,condition,stage,mode),rs in groups.items():
        if mode!='sustained' or not condition.startswith('patch_'): continue
        controls={r['case']:r for r in groups[(suite,condition,stage,'single')]}
        ds=[int(r['both'])-int(controls[r['case']]['both']) for r in rs]
        boot=sorted(sum(rng.choices(ds,k=len(ds)))/len(ds) for _ in range(10000))
        contrasts.append(dict(suite=suite,condition=condition,stage=stage,n=len(ds),
                              sustained_minus_single=sum(ds)/len(ds),paired_bootstrap95=[boot[249],boot[9749]]))
    result=dict(status='audited',manifest=m,rows=len(rows),nfe=nfe,self_patches_exact=selfs,
                analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                results_sha256=hashlib.sha256((root/'results.jsonl').read_bytes()).hexdigest(),summary=summary,contrasts=contrasts,
                warning='Bootstrap intervals with all-equal effects are degenerate and do not establish population certainty.')
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('AUDIT OK',len(rows),'rows',nfe,'forwards',selfs,'self controls')
    for r in summary:
        if r['condition'].startswith(('patch_','token_','generated_')):print(json.dumps(r))


if __name__=='__main__': main()
