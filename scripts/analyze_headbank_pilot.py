import argparse,hashlib,json
from pathlib import Path
from analyze_mechanism_pilot import oracle

def analyze(root,oldroot):
    m=json.loads((root/'manifest.json').read_text());assert m['status']=='complete'
    rows=[json.loads(x) for x in (root/'results.jsonl').read_text().splitlines()]
    old=[json.loads(x) for x in (oldroot/'results.jsonl').read_text().splitlines()]
    baselines={(r['case'],r['variant']):r for r in old if r['kind']=='baseline'}
    baselines.update({(r['case'],r['variant']):r for r in rows if r['phase']=='baseline'})
    cases={c['id']:c for c in m['data']};summary={};keys=[];self_count=0
    assert len(rows)==m['rows']==162
    for r in rows:
        c=cases[r['case']];isbase=r['phase']=='baseline'
        v=r['variant'] if isbase else ('base' if r['control'] else r['target'])
        expected=oracle(c['variants'][v]);assert expected==r['expected']
        assert r['correct']==all(r['values'][k]==expected[k] for k in 'AB')
        state=r['initial_state'].copy()
        for w in r['writes']:
            assert state[w['position']]==5
            best=max(w['predictions'],key=lambda x:(x['confidence'],-x['position']))
            assert (best['position'],best['token_id'])==(w['position'],w['token_id'])
            if not isbase:
                assert w['recipient_state']==state;d=w['donor_state']
                assert [x==5 for x in d]==[x==5 for x in state]
                assert all(d[p]==state[p] for k in 'AB' for p in r['spans'][k])
                if r['control']=='base':assert d==state
            state[w['position']]=w['token_id']
        assert state==r['ids']
        for k in 'AB':
            text=''.join(next(w['token_text'] for w in r['writes'] if w['position']==p) for p in r['spans'][k])
            assert (int(text) if len(text)==2 and text.isdecimal() else None)==r['values'][k]
        if not isbase:
            donor=r['control'] or r['target'];base=baselines[(r['case'],'base')]
            assert r['baseline_values']==base['values']
            assert r['eligible']==all(baselines[(r['case'],v)]['values']==oracle(c['variants'][v]) for v in ('base',donor))
            assert r['induced']==(r['correct'] and any(r['baseline_values'][k]!=expected[k] for k in 'AB'))
            assert len(r['writes'])==4 and r['nfe']==8
            assert r['size']==sum(len(hh) for hh in r['bank'].values())
            if r['control']=='base':assert r['ids']==base['ids'];self_count+=1
            if r['name']=='all_layers' and r['control'] is None:
                assert all(r['values'][k]==baselines[(r['case'],donor)]['values'][k] for k in 'AB')
        key=(r['split'],r['phase'],r.get('name'),r.get('target'),r.get('control'),r.get('variant'))
        keys.append((r['case'],*key))
        if key not in summary:summary[key]=dict(split=r['split'],phase=r['phase'],name=r.get('name'),target=r.get('target'),control=r.get('control'),variant=r.get('variant'),size=r.get('size'),n=0,correct=0,induced=0,eligible=0,eligible_induced=0,other_unchanged=0,changed=0)
        s=summary[key];s['n']+=1;s['correct']+=r['correct'];s['induced']+=r.get('induced',False);s['eligible']+=r.get('eligible',False);s['eligible_induced']+=r.get('induced',False) and r.get('eligible',False)
        if not isbase:
            s['changed']+=any(r['values'][k]!=r['baseline_values'][k] for k in 'AB')
            if r['target'] in ('U','V'):
                other='B' if r['target']=='U' else 'A';s['other_unchanged']+=r['values'][other]==r['baseline_values'][other]
    assert len(keys)==len(set(keys));assert sum(r['nfe'] for r in rows)==m['nfe']
    for phase,selected in [('layer_discovery',m['layer_choice']),('head_subset_discovery',m['subset_choice'])]:
        candidates=[s for s in summary.values() if s['phase']==phase]
        best=max(candidates,key=lambda s:(s['eligible_induced'],s['correct'],-s['size'],s['name']))
        assert best['name']==selected
    return dict(audited=True,rows=len(rows),nfe=m['nfe'],self_controls=self_count,layer_choice=m['layer_choice'],subset_choice=m['subset_choice'],selected=m['selected'],summary=list(summary.values()),results_sha256=hashlib.sha256((root/'results.jsonl').read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('previous');p.add_argument('--out',required=True);a=p.parse_args()
    s=analyze(Path(a.run),Path(a.previous));Path(a.out).write_text(json.dumps(s,indent=2),encoding='utf-8')
    print('HEADBANK AUDIT PASSED',s['rows'],s['nfe'],'self controls',s['self_controls'])
    print('BANK',s['layer_choice'],s['subset_choice'])
    for r in s['summary']:print(json.dumps(r))
