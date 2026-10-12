"""Small paired single/joint generation screen; no gold-dependent output slots."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dlm_coupling.backends.hf import HFBackend
from dlm_coupling.tasks.igsm import parse_answers, render, validate_pair
from dlm_coupling.text_generation import generate_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['dream', 'illada'], required=True)
    p.add_argument('--data', default='data/generated/igsm-pairs-v1.jsonl')
    p.add_argument('--out', required=True)
    p.add_argument('--single-budget', type=int, default=64)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists(): raise ValueError('Use a fresh output directory')
    pairs = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines()]
    for pair in pairs: validate_pair(pair)
    if len({x['facts_sha256'] for x in pairs}) != len(pairs): raise ValueError('Repeated source graph')
    revision = {'dream': '05334cb9faaf763692dcf9d8737c642be2b2a6ae',
                'illada': '5769f049224b2cff61106ebce0429ede5deeb434'}[args.model]
    config = json.loads((ROOT/f'configs/models/{args.model}.json').read_text(encoding='utf-8'))
    config['revision'] = revision
    out.mkdir(parents=True)
    manifest = {'status': 'starting', 'model_config': config, 'single_budget': args.single_budget,
                'joint_budget': 2*args.single_budget, 'tokens_per_step': 1,
                'source_digest': hashlib.sha256(b''.join(f.read_bytes() for f in sorted((ROOT/'src').rglob('*.py')))).hexdigest(),
                'dataset_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                'n_pairs': len(pairs), 'expected_generations': 4*len(pairs),
                'protocol': 'Single A, single B, joint AB, joint BA; same facts; strict JSON; modulo 23'}
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save()
    rows = []
    try:
        backend = HFBackend(config)
        # Verify representation only; do not demand any single-token integers.
        audit = {str(v): backend.tokenizer.encode(str(v), add_special_tokens=False) for v in range(23)}
        (out/'tokenizer_numbers.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
        backend._load_model()
        manifest['status'] = 'running'; save()
        with (out/'results.jsonl').open('w',encoding='utf-8') as stream:
            for pair in pairs:
                for order in ('A', 'B', 'AB', 'BA'):
                    prompt = render(pair,order)
                    start = time.perf_counter()
                    trace = generate_text(backend,prompt,args.single_budget*len(order))
                    elapsed = time.perf_counter()-start
                    answer,error = parse_answers(trace['text'],order)
                    row = {'pair_id':pair['pair_id'],'relation':pair['relation'],'order':order,
                           'prompt':prompt,'gold':{q:pair['targets'][q]['gold'] for q in order},
                           'answers':answer,'parse_error':error,'trace':trace,'seconds':elapsed,
                           'correct':{q:answer.get(q)==pair['targets'][q]['gold'] for q in order}}
                    stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
                    print(args.model,pair['relation'],pair['pair_id'],order,repr(trace['text']),row['correct'],flush=True)
        summary=[]
        for relation in ('independent','shared','downstream'):
            for mode in ('single','joint'):
                selected=[r for r in rows if r['relation']==relation and (len(r['order'])==1)==(mode=='single')]
                n=sum(len(r['correct']) for r in selected)
                summary.append({'relation':relation,'mode':mode,'correct':sum(sum(r['correct'].values()) for r in selected),
                                'answers':n,'format_failures':sum(r['parse_error'] is not None for r in selected),
                                'generations':len(selected),'independent_pairs':len({r['pair_id'] for r in selected})})
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        manifest.update(status='complete',completed_generations=len(rows),backend=backend.metadata,
                        experimental_nfe=sum(r['trace']['nfe'] for r in rows))
        save();print('COMPLETE',args.model,json.dumps(summary),flush=True)
    except BaseException as exc:
        manifest.update(status='failed',error=str(exc),completed_generations=len(rows));save();raise


if __name__=='__main__': main()
