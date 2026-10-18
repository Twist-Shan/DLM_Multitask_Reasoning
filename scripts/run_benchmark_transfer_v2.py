"""Bounded MuSiQue/StepGame calibration with predeclared decoding controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dlm_coupling import benchmark_transfer as base, benchmark_transfer_v2 as task
from run_precommit import MODELS


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def make_jobs(data, per_cell):
    cases = task.selected_cases(data, per_cell)
    jobs = []
    for c in cases:
        for requested in ('B','AB','BA','RB'):
            jobs.append(dict(case_id=c['id'], benchmark=c['benchmark'], hops=c['hops'],
                             context='full', requested=requested, sampler='block32',
                             prompt=task.prompt(c,requested)))
    for benchmark in ('musique','stepgame'):
        c = next(c for c in cases if c['benchmark']==benchmark and c['hops']==2)
        for requested in ('B','AB'):
            jobs.append(dict(case_id=c['id'], benchmark=benchmark, hops=2, context='full',
                             requested=requested, sampler='full', prompt=task.prompt(c,requested)))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--per-cell',type=int,default=3)
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--budget',type=int,default=1024)
    ap.add_argument('--max-seconds',type=int,default=3600)
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()
    if args.budget<1 or args.max_seconds<1:
        ap.error('Positive budget/time required')
    args.out.mkdir(parents=True,exist_ok=False)
    data=json.loads(args.dataset.read_text(encoding='utf-8'))
    jobs=make_jobs(data,args.per_cell)
    cfg=dict(**MODELS['illada'],dtype='bfloat16',device='cuda',trust_remote_code=True)
    manifest=dict(protocol=task.PROTOCOL,status='prepared',seed=args.seed,model=cfg,
                  dataset_sha256=base.digest(data),jobs=len(jobs),rows=0,
                  budget=args.budget,tokens_per_step=4,block_size=32,per_cell=args.per_cell,
                  compute=dict(forwards=0,processed_token_positions=0),
                  code_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(__file__),*sorted((ROOT/'src').rglob('*.py'))]})
    dump(args.out/'dataset.json',data)
    dump(args.out/'jobs.json',jobs)
    dump(args.out/'manifest.json',manifest)
    if args.prepare_only:
        print(json.dumps(dict(jobs=len(jobs),dataset_sha256=base.digest(data))))
        return
    from dlm_coupling.backends.hf import HFBackend
    from dlm_coupling.text_generation import generate_text,generate_blockwise
    backend=HFBackend(cfg)
    limits=[getattr(backend.model_config,k,None) for k in ('max_position_embeddings','max_sequence_length','max_seq_len')]
    limit=min([v for v in limits if type(v) is int and v>0] or [8192])
    for j in jobs:
        j['prompt_tokens']=len(backend.tokenizer.apply_chat_template([dict(role='user',content=j['prompt'])],tokenize=True,add_generation_prompt=True))
    excluded={j['case_id'] for j in jobs if j['prompt_tokens']+args.budget>limit}
    for j in jobs:
        j['eligible']=j['case_id'] not in excluded
    dump(args.out/'jobs.json',jobs)
    manifest.update(status='running',context_limit=limit,excluded_case_ids=sorted(excluded))
    dump(args.out/'manifest.json',manifest)
    original=backend.predict
    def counted(ids,positions,probes):
        result=original(ids,positions,probes)
        manifest['compute']['forwards']+=1
        manifest['compute']['processed_token_positions']+=len(ids)
        return result
    backend.predict=counted
    by_id={c['id']:c for c in data['cases']}
    started=time.monotonic()
    try:
        backend._load_model()
        backend._torch.manual_seed(args.seed)
        for job in jobs:
            if time.monotonic()-started>args.max_seconds:
                manifest['status']='time_limit_partial'
                break
            if not job['eligible']:
                continue
            before=dict(manifest['compute'])
            tick=time.monotonic()
            if job['sampler']=='block32':
                generation=generate_blockwise(backend,job['prompt'],args.budget,32,tokens_per_step=4)
            else:
                generation=generate_text(backend,job['prompt'],args.budget,tokens_per_step=4)
            row=dict(**job,generation=generation,seconds=time.monotonic()-tick,
                     compute={k:manifest['compute'][k]-before[k] for k in before},
                     **task.score(by_id[job['case_id']],generation['text'],job['requested']))
            with (args.out/'results.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            manifest['rows']+=1
            manifest['elapsed_seconds']=time.monotonic()-started
            dump(args.out/'manifest.json',manifest)
            print(json.dumps({k:row[k] for k in ('case_id','sampler','requested','format_valid','scores','seconds')}),flush=True)
        else:
            manifest['status']='complete'
    except Exception as exc:
        manifest.update(status='failed_partial',error=repr(exc))
        raise
    finally:
        manifest['elapsed_seconds']=time.monotonic()-started
        dump(args.out/'manifest.json',manifest)


if __name__=='__main__':
    main()
