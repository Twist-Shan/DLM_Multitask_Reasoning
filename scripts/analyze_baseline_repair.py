"""Independent arithmetic and token-write audit of the repair experiment."""
import argparse,hashlib,json,re
from pathlib import Path

def analyze(root):
    m=json.loads((root/'manifest.json').read_text());assert m['status']=='complete'
    rows=[json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
    cases={c['id']:c for items in m['data'].values() for c in items}
    assert len(rows)==80 and len({(r['case'],r['variant']) for r in rows})==80
    parser_differences=[]
    for r in rows:
        c=cases[r['case']];p=c['base'];C=p['x']+p['y']
        A=C+p['u']+(p['r'] if c['suite']=='branches' else 0)
        B=2*C+p['v']+(p['s'] if c['suite']=='branches' else 0)
        assert (r['expected']['A'],r['expected']['B'])==(A,B)
        assert r['correct']==(r['values']['A']==A and r['values']['B']==B)
        if r['variant']=='natural':
            text=r['generation']['text'].rsplit('</think>',1)[-1]
            matches=re.findall(r'^FINAL: A\s*=\s*(\d+)\s*;\s*B\s*=\s*(\d+)\s*\.?\s*$',text,re.M)
            strict=dict(zip('AB',map(int,matches[0]))) if len(matches)==1 else dict(A=None,B=None)
            raw_matches=re.findall(r'FINAL: A\s*=\s*(\d+)\s*;\s*B\s*=\s*(\d+)\s*\.?',r['generation']['text'])
            online=dict(zip('AB',map(int,raw_matches[0]))) if len(raw_matches)==1 else dict(A=None,B=None)
            assert (strict if m.get('parser_version',1)>=2 else online)==r['values']
            if strict!=online: parser_differences.append(dict(case=r['case'],online=online,strict=strict))
            r['strict_correct']=strict==dict(A=A,B=B)
            assert r['nfe']==len(r['generation']['steps'])
        else:
            state=r['initial_state'].copy()
            for w in r['writes']:
                assert state[w['position']]==5
                pred=max(w['predictions'],key=lambda x:(x['confidence'],-x['position']))
                assert (pred['position'],pred['token_id'])==(w['position'],w['token_id'])
                state[w['position']]=w['token_id']
            assert state==r['ids'] and len(r['writes'])==r['nfe']
            for key in 'AB':
                text=''.join(next(w['token_text'] for w in r['writes'] if w['position']==pos) for pos in r['spans'][key])
                assert text==r['texts'][key]
                assert (int(text) if re.fullmatch(r'\d{2}',text) else None)==r['values'][key]
    assert sum(r['nfe'] for r in rows)==m['nfe']
    summary=[]
    for s in m['summary']:
        g=[r for r in rows if (r['split'],r['suite'],r['variant'])==(s['split'],s['suite'],s['variant'])]
        assert (len(g),sum(r['correct'] for r in g),sum(r['nfe'] for r in g))==(s['n'],s['correct'],s['nfe'])
        summary.append(dict(**s,strict_correct=sum(r.get('strict_correct',r['correct']) for r in g),example_answer_matches=sum((r['values']['A'],r['values']['B'])==((12,23) if s['suite']=='replication' else (22,34)) for r in g)))
    for suite,v in m['chosen'].items():
        g=[s for s in summary if s['split']=='calibration' and s['suite']==suite and s['variant']!='old_masked']
        expected=max(g,key=lambda s:(s['correct'],-s['nfe'],-m['variants'].index(s['variant'])))['variant']
        assert expected==v
        strict_selected=max(g,key=lambda s:(s['strict_correct'],-s['nfe'],-m['variants'].index(s['variant'])))['variant']
        assert strict_selected==v,'Parser correction changes selection; fresh heldout needed'
    return dict(summary=summary,chosen=m['chosen'],rows=len(rows),nfe=m['nfe'],audited=True,parser_differences=parser_differences,
                results_sha256=hashlib.sha256((root/'results.jsonl').read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--out',required=True);a=p.parse_args()
    result=analyze(Path(a.run));Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
