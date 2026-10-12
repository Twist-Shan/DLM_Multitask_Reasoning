"""Accuracy calibration on explicit ordinary-integer iGSM subgraphs."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling.backends.hf import HFBackend
from dlm_coupling.tasks.igsm_explicit import parse_answers,render,validate_pair
from dlm_coupling.text_generation import generate_text,generate_blockwise


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['dream','illada'],required=True)
    ap.add_argument('--data',default='data/generated/igsm-explicit-cal-v2.jsonl')
    ap.add_argument('--out',required=True)
    ap.add_argument('--single-budget',type=int,default=512)
    ap.add_argument('--tokens-per-step',type=int,default=8)
    ap.add_argument('--prompt-version',choices=['v1','v2'],default='v2')
    ap.add_argument('--sampler',choices=['full','blockwise'],default='blockwise')
    ap.add_argument('--block-size',type=int,default=32)
    ap.add_argument('--seed',type=int,default=123)
    args=ap.parse_args()
    assert args.single_budget>0 and args.tokens_per_step>0
    out=Path(args.out)
    if out.exists(): raise ValueError('Use a fresh output directory')
    pairs=[json.loads(l) for l in Path(args.data).read_text(encoding='utf-8').splitlines()]
    for pair in pairs: validate_pair(pair)
    assert len({p['facts_sha256'] for p in pairs})==len(pairs)
    config=json.loads((ROOT/f'configs/models/{args.model}.json').read_text())
    config['revision']={'dream':'05334cb9faaf763692dcf9d8737c642be2b2a6ae',
                        'illada':'5769f049224b2cff61106ebce0429ede5deeb434'}[args.model]
    out.mkdir(parents=True)
    manifest={'status':'starting','model_config':config,'args':vars(args),'n_pairs':len(pairs),
              'dataset_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
              'source_digest':hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src').rglob('*.py')))).hexdigest(),
              'protocol':'Explicit equations; ordinary integers; brief reasoning + FINAL_JSON; sampler recorded in args and trace; no gold in generation'}
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows=[];save()
    try:
        random.seed(args.seed)
        backend=HFBackend(config,load_model=True)
        backend._torch.manual_seed(args.seed)
        backend._torch.cuda.manual_seed_all(args.seed)
        manifest['status']='running';save()
        with (out/'results.jsonl').open('w',encoding='utf-8') as stream:
            for pair in pairs:
                for order in ('A','B','AB','BA'):
                    prompt=render(pair,order,version=args.prompt_version)
                    start=time.perf_counter()
                    trace=(generate_blockwise(backend,prompt,args.single_budget*len(order),args.block_size)
                           if args.sampler=='blockwise' else generate_text(backend,prompt,args.single_budget*len(order),args.tokens_per_step))
                    ans,error=parse_answers(trace['text'],order)
                    row={'pair_id':pair['pair_id'],'relation':pair['relation'],'order':order,
                         'prompt':prompt,'gold':{q:pair['targets'][q]['gold'] for q in order},
                         'answers':ans,'parse_error':error,'trace':trace,'seconds':time.perf_counter()-start,
                         'correct':{q:ans.get(q)==pair['targets'][q]['gold'] for q in order}}
                    stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
                    print(args.model,pair['pair_id'],order,'answer',ans,'error',error,'correct',row['correct'],
                          'length',trace['effective_output_tokens'],'EOS',trace['eos_found'],flush=True)
                    if error: print('RAW_TAIL',repr(trace['text'][-600:]),flush=True)
        summary={}
        for mode,orders in [('single',('A','B')),('AB',('AB',)),('BA',('BA',))]:
            rs=[r for r in rows if r['order'] in orders]
            summary[mode]={'correct':sum(sum(r['correct'].values()) for r in rs),
                           'answers':sum(len(r['correct']) for r in rs),
                           'parse_failures':sum(r['parse_error'] is not None for r in rs),
                           'no_eos':sum(not r['trace']['eos_found'] for r in rs),'generations':len(rs)}
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        manifest.update(status='complete',completed_generations=len(rows),backend=backend.metadata,
                        experimental_nfe=sum(r['trace']['nfe'] for r in rows))
        save();print('COMPLETE',args.model,json.dumps(summary),flush=True)
    except BaseException as e:
        manifest.update(status='failed',error=str(e),completed_generations=len(rows));save();raise

if __name__=='__main__': main()
