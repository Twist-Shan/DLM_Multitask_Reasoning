"""Recompute scores and validate every write in downloaded explicit-task runs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling.tasks.igsm_explicit import parse_answers,render,validate_pair


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--artifacts',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    root=Path(args.artifacts);dest=Path(args.out)
    if dest.exists(): raise ValueError('Use a new output file')
    summaries=[];all_rows={}
    for folder in sorted((root/'runs').iterdir()):
        if not (folder/'manifest.json').exists(): continue
        m=json.loads((folder/'manifest.json').read_text())
        assert m['status']=='complete',folder
        if m['args'].get('prompt_version','v1')=='v1':
            with zipfile.ZipFile(root/'runs/source-v1.zip') as z:
                source=b''.join(z.read(n) for n in sorted(z.namelist()) if n.startswith('src/') and n.endswith('.py'))
        else:
            source=b''.join(p.read_bytes() for p in sorted((root/'src').rglob('*.py')))
        assert hashlib.sha256(source).hexdigest()==m['source_digest'],folder
        data=root/m['args']['data']
        assert hashlib.sha256(data.read_bytes()).hexdigest()==m['dataset_sha256']
        pairs=[json.loads(l) for l in data.read_text().splitlines()]
        for pair in pairs: validate_pair(pair)
        byid={p['pair_id']:p for p in pairs}
        rows=[json.loads(l) for l in (folder/'results.jsonl').read_text().splitlines()]
        all_rows[folder.name]={(r['pair_id'],r['order']):r for r in rows}
        assert len(rows)==m['completed_generations']==4*len(pairs)
        assert Counter((r['pair_id'],r['order']) for r in rows)==Counter((p['pair_id'],o) for p in pairs for o in ('A','B','AB','BA'))
        for r in rows:
            pair=byid[r['pair_id']];o=r['order'];t=r['trace']
            assert r['prompt']==render(pair,o,version=m['args'].get('prompt_version','v1')) and r['relation']==pair['relation']
            gold={q:pair['targets'][q]['gold'] for q in o}
            assert gold==r['gold']
            ans,error=parse_answers(t['text'],o)
            assert (ans,error)==(r['answers'],r['parse_error'])
            assert r['correct']=={q:ans.get(q)==gold[q] for q in o}
            assert t['output_budget']==m['args']['single_budget']*len(o)
            actual=len(t['output_token_ids'])
            if m['args'].get('sampler','full')=='blockwise':
                assert t['nfe']==len(t['steps'])==actual<=t['output_budget']
            else:
                assert t['nfe']==len(t['steps'])==math.ceil(t['output_budget']/m['args']['tokens_per_step'])
            writes=[w for s in t['steps'] for w in s['writes']]
            assert Counter(w['output_position'] for w in writes)==Counter(range(actual))
            assert all(t['output_token_ids'][w['output_position']]==w['token_id'] for w in writes)
            assert not t['gold_used_for_generation']
        assert sum(r['trace']['nfe'] for r in rows)==m['experimental_nfe']
        for rel in ('all','independent','shared','downstream'):
            for mode,orders in [('single',('A','B')),('AB',('AB',)),('BA',('BA',))]:
                rs=[r for r in rows if r['order'] in orders and (rel=='all' or r['relation']==rel)]
                if not rs: continue
                ids={r['pair_id'] for r in rs}
                s={'run':folder.name,'relation':rel,'mode':mode,'correct':sum(sum(r['correct'].values()) for r in rs),
                   'answers':sum(len(r['correct']) for r in rs),'pairs':len(ids),
                   'both_correct':sum(all(all(r['correct'].values()) for r in rs if r['pair_id']==pid) for pid in ids),
                   'parse_failures':sum(r['parse_error'] is not None for r in rs),'generations':len(rs),
                   'no_eos':sum(not r['trace']['eos_found'] for r in rs),
                   'total_nfe':sum(r['trace']['nfe'] for r in rs),
                   'mean_effective_tokens':sum(r['trace']['effective_output_tokens'] for r in rs)/len(rs),
                   'generation_seconds':sum(r['seconds'] for r in rs),
                   'max_effective_tokens':max(r['trace']['effective_output_tokens'] for r in rs)}
                summaries.append(s)
                if rel=='all': print(json.dumps(s))
    for family in ('dream','illada'):
        short=all_rows.get('calibration-v2-'+family)
        long=all_rows.get('calibration-v2-long-'+family)
        if short is not None and long is not None:
            assert set(short)==set(long)
            for key,r in short.items():
                old=r['trace']['output_token_ids'];new=long[key]['trace']['output_token_ids']
                assert new[:len(old)]==old,(family,key,'Budget changed original token prefix')
            print('BUDGET PREFIX IDENTICAL',family,len(short))
    assert summaries,'No completed runs'
    dest.parent.mkdir(exist_ok=True,parents=True)
    dest.write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    print('AUDIT PASSED')

if __name__=='__main__': main()
