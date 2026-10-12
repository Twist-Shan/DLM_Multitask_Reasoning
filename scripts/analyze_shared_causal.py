"""Audit and summarize fixed-layout pilot, counting A/B separately from C."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();root=Path(a.run)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='complete'
    rows=[json.loads(s) for s in (root/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==len(manifest['cases'])*(7+8*len(manifest['layers']))
    bases={(r['case'],r['stage']):r for r in rows if r['condition']=='patch_baseline'}
    groups=defaultdict(list)
    for r in rows:
        groups[(r['condition'],r.get('stage',''),r.get('layer',-1))].append(r)
        if r['condition']=='self': assert r['ids']==bases[(r['case'],r['stage'])]['ids']
        # Every continuation should commit each A/B slot once, without rewriting.
        positions=[w['position'] for w in r['writes']]
        assert len(positions)==len(set(positions))
        assert all(w['token_id']!=5 for w in r['writes'])
    summary=[]
    for (condition,stage,layer),rs in groups.items():
        both=valid=unchanged=first_both=0
        for r in rs:
            expected=r.get('expected',r.get('gold',manifest['cases'][r['case']]['gold']))
            values=r['result']['values']
            valid+=all(values[k] is not None for k in ('A','B'))
            both+=all(values[k]==expected[k] for k in ('A','B'))
            if 'stage' in r:
                unchanged+=all(values[k]==bases[(r['case'],stage)]['result']['values'][k] for k in ('A','B'))
            # Immediate per-position argmax, not a joint likelihood or rollout.
            if condition not in ('generated_C','donor_generated_C','same_generated_C'):
                predictions=sorted(r['writes'][0]['predictions'],key=lambda p:p['position'])
                text=''.join(p['token_text'] for p in predictions)
                first_both+=text==str(expected['A'])+str(expected['B'])
        summary.append(dict(condition=condition,stage=stage,layer=layer,n=len(rs),
                            valid_AB=valid,expected_both=both,unchanged_AB=unchanged,immediate_argmax_expected_both=first_both))
    result=dict(manifest=manifest,rows=len(rows),results_sha256=hashlib.sha256((root/'results.jsonl').read_bytes()).hexdigest(),summary=summary,
                notes=['Exploratory layer scan on six cases; no held-out confirmation.',
                       'C_masked retains masked C while filling A/B; validity and accuracy refer to A/B.',
                       'C_visible_oracle supplies ground-truth intermediate C, never supplies A/B.',
                       'Patch affects one forward; later forwards recompute from recipient tokens.',
                       'Unrelated-position control is adjacent non-numeral syntax; its true relevance is unknown.'])
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    for item in summary: print(json.dumps(item))


if __name__=='__main__': main()
