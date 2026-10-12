"""Validate downloaded iGSM traces and report paired descriptive results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dlm_coupling.tasks.igsm import parse_answers, render, validate_pair


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--artifacts',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    root=Path(args.artifacts)
    data=root/'data/generated/igsm-pairs-v1.jsonl'
    pairs=[json.loads(line) for line in data.read_text(encoding='utf-8').splitlines()]
    for pair in pairs: validate_pair(pair)
    lookup={p['pair_id']:p for p in pairs}
    output=[]
    for family in ('dream','illada'):
        folder=root/f'runs/igsm-{family}'
        m=json.loads((folder/'manifest.json').read_text())
        assert m['status']=='complete' and m['n_pairs']==len(pairs)
        assert m['dataset_sha256']==hashlib.sha256(data.read_bytes()).hexdigest()
        rows=[json.loads(line) for line in (folder/'results.jsonl').read_text().splitlines()]
        assert len(rows)==4*len(pairs)==m['completed_generations']
        assert Counter((r['pair_id'],r['order']) for r in rows)==Counter((p['pair_id'],o) for p in pairs for o in ('A','B','AB','BA'))
        for row in rows:
            pair=lookup[row['pair_id']]; order=row['order']; t=row['trace']
            assert row['relation']==pair['relation'] and row['prompt']==render(pair,order)
            gold={q:pair['targets'][q]['gold'] for q in order}
            assert row['gold']==gold
            ans,error=parse_answers(t['text'],order)
            assert ans==row['answers'] and error==row['parse_error']
            assert row['correct']=={q:ans.get(q)==gold[q] for q in order}
            # Supplementary per-field diagnostic: the frozen primary parser
            # rejects the entire response if any field has an invalid domain.
            diagnostic = ans
            if error == 'domain':
                text=t['text'].strip()
                if text.startswith('```'):
                    text=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',text,re.DOTALL).group(1)
                diagnostic=json.loads(text)
            row['diagnostic_correct']={q:type(diagnostic.get(q)) is int and diagnostic[q]==gold[q] for q in order}
            row['domain_invalid_fields']=sum(type(v) is not int or not 0<=v<23 for v in diagnostic.values())
            assert t['nfe']==len(t['steps'])==t['output_budget']==m['single_budget']*len(order)
            assert not t['gold_used_for_generation']
            written={}
            for step in t['steps']:
                assert len(step['writes'])==1
                w=step['writes'][0]; pos=w['output_position']
                assert pos not in written and 0<=pos<t['output_budget']
                written[pos]=w['token_id']
            assert [written[i] for i in range(t['output_budget'])]==t['output_token_ids']
        assert sum(r['trace']['nfe'] for r in rows)==m['experimental_nfe']
        for relation in ('all','independent','shared','downstream'):
            selected=[r for r in rows if relation=='all' or r['relation']==relation]
            by={(r['pair_id'],r['order']):r for r in selected}
            for mode,orders in [('single',('A','B')),('AB',('AB',)),('BA',('BA',))]:
                rr=[r for r in selected if r['order'] in orders]
                ids=sorted({r['pair_id'] for r in rr})
                both=sum(all(all(by[(pid,o)]['correct'].values()) for o in orders) for pid in ids)
                output.append({'model':family,'relation':relation,'mode':mode,
                    'correct':sum(sum(r['correct'].values()) for r in rr),
                    'answers':sum(len(r['correct']) for r in rr),
                    'both_correct':both,'source_pairs':len(ids),
                    'strict_valid':sum(r['parse_error'] is None for r in rr),'generations':len(rr),
                    'json_and_keys_valid':sum(r['parse_error'] in (None,'domain') for r in rr),
                    'domain_invalid_fields':sum(r['domain_invalid_fields'] for r in rr),
                    'diagnostic_field_correct':sum(sum(r['diagnostic_correct'].values()) for r in rr)})
    dest=Path(args.out)
    if dest.exists(): raise ValueError('Use a new summary output')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(output,indent=2),encoding='utf-8')
    for row in output: print(json.dumps(row))
    print('VALIDATED',len(pairs),'source pairs per model')


if __name__=='__main__': main()
