"""Recompute arithmetic, replay token writes, and summarize every condition."""
import argparse,hashlib,json
from pathlib import Path

def oracle(p):
    c=p['x']+p['y'];u=p['u']+p['r'];v=p['v']+p['s']
    return dict(C=c,U=u,V=v,A=c+u,B=2*c+v)

def analyze(root):
    m=json.loads((root/'manifest.json').read_text());assert m['status']=='complete'
    rows=[json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
    cases={c['id']:c for c in m['data']};assert len(rows)==m['rows']
    keys=[];summaries={};controls=0
    baselines={(r['case'],r['variant']):r for r in rows if r['kind']=='baseline'}
    for r in rows:
        c=cases[r['case']];key=(r['case'],r['kind'],r.get('variant'),r.get('config'),r.get('target'),r.get('control'),r.get('mode'),r.get('head'),r.get('stage'))
        keys.append(key)
        v=r['variant'] if r['kind']=='baseline' else 'base' if r['control'] else r['target']
        expected=oracle(c['variants'][v]);assert expected==r['expected']
        assert r['correct']==all(r['values'][k]==expected[k] for k in 'AB')
        state=r['initial_state'].copy()
        for w in r['writes']:
            assert state[w['position']]==5
            chosen=max(w['predictions'],key=lambda x:(x['confidence'],-x['position']))
            assert (chosen['position'],chosen['token_id'])==(w['position'],w['token_id'])
            if w.get('patched'):
                d=w['donor_state'];assert w['recipient_state']==state and len(d)==len(state)
                assert [x==5 for x in d]==[x==5 for x in state]
                assert all(d[p]==state[p] for k in 'AB' for p in r['spans'][k])
                if r['control']=='base':assert d==state
            state[w['position']]=w['token_id']
        assert state==r['ids']
        for k in 'AB':
            text=''.join(next(w['token_text'] for w in r['writes'] if w['position']==p) for p in r['spans'][k])
            assert r['values'][k]==(int(text) if len(text)==2 and text.isdecimal() else None)
        if r['kind']!='baseline':
            source=r['control'] if r['control'] in ('base','same') else r['target']
            eligible=all(baselines[(r['case'],v)]['values']==oracle(c['variants'][v]) for v in ('base',source))
            assert r['eligible']==eligible
            assert r['induced']==(r['correct'] and any(r['baseline_values'][k]!=expected[k] for k in 'AB'))
            assert sum(w['patched'] for w in r['writes'])==(1 if r['mode']=='single' else 4)
            assert r['nfe']==4+(1 if r['mode']=='single' else 4)+(4 if r['stage']=='initial' else 0)
            if r['control']=='base':
                assert all(r['values'][k]==r['baseline_values'][k] for k in 'AB');controls+=1
        group=(r['split'],r['kind'],r.get('variant'),r.get('config'),r.get('target'),r.get('control'),r.get('mode'),r.get('head'),r.get('stage'))
        if group not in summaries:
            summaries[group]=dict(split=r['split'],kind=r['kind'],variant=r.get('variant'),config=r.get('config'),target=r.get('target'),control=r.get('control'),mode=r.get('mode'),head=r.get('head'),stage=r.get('stage'),n=0,correct=0,induced=0,eligible=0,eligible_induced=0,other_unchanged=0,new_A=0,new_B=0,changed_pairs=0,baseline_correct=0)
        s=summaries[group];s['n']+=1;s['correct']+=r['correct'];s['induced']+=r.get('induced',False)
        s['eligible']+=r.get('eligible',False);s['eligible_induced']+=r.get('eligible',False) and r.get('induced',False)
        if r['kind']!='baseline':
            s['baseline_correct']+=all(r['baseline_values'][k]==oracle(c['variants']['base'])[k] for k in 'AB')
            for k in 'AB':s['new_'+k]+=r['values'][k]==expected[k] and r['baseline_values'][k]!=expected[k]
            s['changed_pairs']+=any(r['values'][k]!=r['baseline_values'][k] for k in 'AB')
        if r.get('target') in ('U','V'):
            other='B' if r['target']=='U' else 'A';s['other_unchanged']+=r['values'][other]==r['baseline_values'][other]
    assert len(keys)==len(set(keys));assert sum(r['nfe'] for r in rows)+m['representation_forwards']==m['nfe']
    assert len([r for r in rows if r['kind']=='grid'])==96
    assert len([r for r in rows if r['kind']=='head_discovery'])==128
    assert len([r for r in rows if r['kind']=='baseline'])==50
    # Recompute frozen choices without importing selection implementation.
    grid=[s for s in summaries.values() if s['kind']=='grid']
    def rank(s):return (s['eligible_induced'],s['correct'],-int(s['config'].split('-L')[1].split('-')[0]),s['config'])
    assert max(grid,key=rank)['config']==m['chosen']
    assert max([s for s in grid if s['config'].startswith('attention-') and s['config'].endswith('-AB')],key=rank)['config']==m['attention_chosen']
    heads=[s for s in summaries.values() if s['kind']=='head_discovery']
    best=max(heads,key=lambda s:(s['eligible_induced'],s['correct'],-s['head']))
    assert (best['head'] if best['eligible_induced'] else None)==m['head_chosen']
    return dict(audited=True,rows=len(rows),nfe=m['nfe'],self_controls=controls,chosen=m['chosen'],attention_chosen=m['attention_chosen'],head_chosen=m['head_chosen'],summary=list(summaries.values()),
                results_sha256=hashlib.sha256((root/'results.jsonl').read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--out',required=True);a=p.parse_args()
    s=analyze(Path(a.run));Path(a.out).write_text(json.dumps(s,indent=2),encoding='utf-8')
    print('AUDIT PASSED',s['rows'],'rows',s['nfe'],'forwards',s['self_controls'],'self controls')
    print('SELECTION',s['chosen'],s['attention_chosen'],'head',s['head_chosen'])
    for x in s['summary']:
        if x['split']=='validation':print(json.dumps(x))
