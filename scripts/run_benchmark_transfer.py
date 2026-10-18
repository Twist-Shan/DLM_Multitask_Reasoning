"""Prepare audited benchmark cases or run bounded free-generation calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dlm_coupling import benchmark_transfer as task


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/external/realistic-v1')
    p.add_argument('--dataset', type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--per-hop', type=int, default=10)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--model', choices=['illada', 'dream'], default='illada')
    p.add_argument('--max-seconds', type=int, default=3600)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--cases-per-cell', type=int, default=3)
    p.add_argument('--hops', nargs='+', type=int, choices=[2,3,4], default=[2,3,4])
    p.add_argument('--requests', nargs='+', choices=['B','AB','BA','RB'], default=['B','AB','BA','RB'])
    p.add_argument('--contexts', nargs='+', choices=['full','support'], default=['full','support'])
    p.add_argument('--max-new-tokens', type=int, default=128)
    args = p.parse_args()
    if args.cases_per_cell < 1 or args.max_new_tokens < 1 or args.per_hop < 1:
        p.error('Sample counts and token budget must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    if args.prepare:
        files = ['musique-dev.jsonl', 'stepgame-valid.jsonl', 'stepgame-semantic-mapping.py']
        rows = [list(map(json.loads, (args.data_dir / f).read_text(encoding='utf-8').splitlines())) for f in files[:2]]
        data = task.prepare(*rows, task.load_mapping(args.data_dir / files[2]), args.per_hop, args.seed)
        data['input_sha256'] = {f: hashlib.sha256((args.data_dir/f).read_bytes()).hexdigest() for f in files}
        dump(args.out / 'dataset.json', data)
        dump(args.out / 'audit.json', data['audit'])
        print(json.dumps(dict(cases=len(data['cases']), audit=data['audit'], sha256=task.digest(data))))
        return
    if not args.dataset:
        p.error('--dataset required for inference')
    data = json.loads(args.dataset.read_text(encoding='utf-8'))
    dump(args.out / 'dataset.json', data)
    from run_precommit import MODELS
    from dlm_coupling.backends.hf import HFBackend
    from dlm_coupling.text_generation import generate_text
    cfg = dict(**MODELS[args.model], dtype='bfloat16', device='cuda', trust_remote_code=True)
    backend = HFBackend(cfg)
    jobs = []
    limits = [getattr(backend.model_config, k, None) for k in
              ('max_position_embeddings', 'max_sequence_length', 'max_seq_len')]
    limit = min([x for x in limits if type(x) is int and x > 0] or [4096])
    counts = {}
    for case in data['cases']:
        if case['hops'] not in args.hops:
            continue
        cell = (case['benchmark'], case['hops'])
        counts[cell] = counts.get(cell, 0) + 1
        if counts[cell] > args.cases_per_cell:
            continue
        for context in (args.contexts if case['benchmark'] == 'musique' else [c for c in args.contexts if c == 'full']):
            group = []
            for requested in args.requests:
                text = task.prompt(case, requested, context)
                tokens = backend.tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True)
                group.append(dict(case_id=case['id'], benchmark=case['benchmark'], hops=case['hops'],
                                  context=context, requested=requested, prompt=text, prompt_tokens=len(tokens)))
            # Paired exclusion: never compare different subsets after context failures.
            eligible = all(j['prompt_tokens'] + args.max_new_tokens <= limit for j in group)
            for j in group:
                j['eligible'] = eligible
            jobs.extend(group)
    dump(args.out / 'jobs.json', jobs)
    manifest = dict(status='tokenizer_audited', model=cfg, dataset_sha256=task.digest(data), seed=args.seed,
                    jobs=len(jobs), eligible=sum(j['eligible'] for j in jobs), context_limit=limit,
                    cases_per_cell=args.cases_per_cell,
                    max_new_tokens=args.max_new_tokens, tokens_per_step=4, rows=0, forwards=0,
                    selected_hops=args.hops, requests=args.requests, contexts=args.contexts,
                    scope='Exploratory adapted validation subsets; no activation patching or official benchmark score')
    dump(args.out / 'manifest.json', manifest)
    if args.audit_only:
        return
    start = time.monotonic()
    try:
        backend._load_model()
        backend._torch.manual_seed(args.seed)
        by_id = {c['id']: c for c in data['cases']}
        for job in jobs:
            if time.monotonic() - start > args.max_seconds:
                manifest['status'] = 'time_limit_partial'
                break
            if not job['eligible']:
                continue
            generation = generate_text(backend, job['prompt'], max_new_tokens=args.max_new_tokens, tokens_per_step=4)
            row = dict(**job, generation=generation, **task.score(by_id[job['case_id']], generation['text'], job['requested']))
            with (args.out / 'results.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            manifest['rows'] += 1
            manifest['forwards'] += generation['nfe']
            print(json.dumps({k: row[k] for k in ('case_id', 'context', 'requested', 'format_valid', 'scores')}), flush=True)
        else:
            manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed_partial', error=repr(exc))
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - start
        dump(args.out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
