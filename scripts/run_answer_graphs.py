"""Prepare or calibrate small answer graphs; no confirmation or patch inference."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dlm_coupling import answer_graphs as task
from run_precommit import MODELS, dump


def conditions(case):
    for field in task.FIELDS:
        yield 'single', field, field, ('confidence',)
    for layout_name, layout in task.layouts(case).items():
        yield layout_name, task.FIELDS, layout, ('confidence', 'fixed_layout')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--per-cell', type=int, default=10)
    parser.add_argument('--families', nargs='+', choices=task.FAMILIES, default=list(task.FAMILIES))
    parser.add_argument('--graphs', nargs='+', choices=task.GRAPHS, default=list(task.GRAPHS))
    parser.add_argument('--model', choices=MODELS, default='illada')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=3600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = task.dataset(args.seed, args.per_cell, args.families, args.graphs)
    dump(args.out / 'dataset.json', data)
    spec = dict(protocol=task.PROTOCOL, seed=args.seed, per_cell=args.per_cell,
                families=args.families, graphs=args.graphs, rows_per_case=9,
                scope='Exploratory calibration only. All generated cases retained; no correctness filtering.',
                orders='Topological/reverse physical layouts, each crossed with confidence/fixed-layout commitment. Prompt definitions independently shuffled.',
                singles='All five definitions present; only requested field and scaffold vary. Joint/single is not a pure slot intervention.',
                canvas='Native chat template and ending, no unused positions; full vocabulary greedy except MASK excluded.',
                sources='Every site: change, same_value, self; exact Target do-intervention oracle. Sources audited but not run.',
                selection='No pass/fail gate or automatic patching. Report per-field, whole-response, t=0 and final accuracy.',
                unit='One generated graph instance. Policies, layouts, answer fields and Source variants are repeated measurements.',
                budgets='Report forwards and processed positions; matching forward counts is not equal FLOPs.')
    dump(args.out / 'specification.json', spec)
    cfg = dict(**MODELS[args.model], dtype='bfloat16', device='cuda', trust_remote_code=True)
    manifest = dict(status='preparing', protocol=task.PROTOCOL, model=cfg, cases=len(data['cases']),
                    dataset_sha256=task.digest(data), specification_sha256=task.digest(spec), rows=0,
                    compute=dict(forwards=0, processed_token_positions=0),
                    code_sha256={str(p.relative_to(ROOT)): __import__('hashlib').sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), *sorted((ROOT / 'src').rglob('*.py'))]})
    started = time.monotonic()
    def save():
        manifest['elapsed_seconds'] = time.monotonic() - started
        dump(args.out / 'manifest.json', manifest)
    def emit(filename, row):
        with (args.out / filename).open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    save()
    if args.prepare_only:
        for case in data['cases']:
            emit('source_pairs.jsonl', dict(case=case['id'], pairs=list(task.source_pairs(case))))
        manifest['status'] = 'prepared_no_model_loaded'; save()
        print('PREPARED', len(data['cases']), 'cases; exact Source/Target oracles generated'); return
    try:
        from dlm_coupling.backends.hf import HFBackend
        backend = HFBackend(cfg)
        builder = task.CanvasBuilder(backend)
        alignments = 0
        max_length = 0
        for case in data['cases']:
            for _, requested, layout, _ in conditions(case):
                cc = builder.build(case, requested, layout)
                max_length = max(max_length, len(cc['ids']))
            # Both joint layouts, every Source and same-value control, before weights.
            pairs = list(task.source_pairs(case))
            for layout in task.layouts(case).values():
                target = builder.build(case, task.FIELDS, layout)
                for pair in pairs:
                    src = builder.build(pair['Source'], task.FIELDS, layout)
                    if len(target['ids']) != len(src['ids']) or target['spans'] != src['spans'] or target['maps'] != src['maps']:
                        raise ValueError(('Source/Target alignment failed', case['id'], pair['site'], pair['kind']))
                    if target['ids'][target['prompt_length']:] != src['ids'][src['prompt_length']:]:
                        raise ValueError('Source changed answer canvas')
                    alignments += 1
            emit('source_pairs.jsonl', dict(case=case['id'], pairs=pairs))
        limits = [getattr(backend.model_config, k, None) for k in ('max_position_embeddings', 'max_sequence_length', 'max_seq_len')]
        limits = [n for n in limits if isinstance(n, int) and n > 0]
        if limits and max_length > min(limits):
            raise ValueError('Context length exceeded')
        manifest.update(tokenizer_alignments=alignments, max_input_tokens=max_length, scaffold_audits=len(builder.templates), backend=backend.metadata)
        save(); print('TOKENIZER_AUDIT', alignments, 'Source/Target pairs;', max_length, 'max tokens', flush=True)
        if args.audit_only:
            manifest['status'] = 'audited_no_model_loaded'; save(); return
        backend._load_model(); backend._torch.manual_seed(args.seed)
        original_predict = backend.predict
        def counted(ids, positions, probes):
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError('Calibration wall-time budget')
            predictions = original_predict(ids, positions, probes)
            manifest['compute']['forwards'] += 1
            manifest['compute']['processed_token_positions'] += len(ids)
            return predictions
        backend.predict = counted
        manifest['status'] = 'running'; save()
        dump(args.out / 'frozen_before_inference.json', dict(specification=spec, manifest=manifest))
        for case in data['cases']:
            expected = task.evaluate(case)
            for layout_name, requested, layout, policies in conditions(case):
                cc = builder.build(case, requested, layout)
                positions = [p for span in cc['spans'].values() for p in span]
                initial = task.predict(backend, cc, cc['ids'], positions)
                initial_ids = cc['ids'].copy()
                for position, pred in zip(positions, initial):
                    initial_ids[position] = pred['token_id']
                t0 = task.read_values(cc, initial_ids)
                for policy in policies:
                    result = task.decode(backend, cc, policy, initial=initial)
                    emit('behavior.jsonl', dict(case=case['id'], family=case['family'], graph=case['graph'],
                         layout_name=layout_name, requested=requested, expected={k: expected[k] for k in requested},
                         canvas={k: v for k, v in cc.items() if k != 'maps'}, value_token_ids=cc['maps'],
                         t0=t0, t0_predictions=initial, **result))
                    manifest['rows'] += 1
            save(); print('CASE_DONE', case['id'], manifest['rows'], flush=True)
        if manifest['rows'] != len(data['cases']) * spec['rows_per_case']:
            raise AssertionError('Incomplete calibration grid')
        manifest.update(status='complete', backend=backend.metadata); save()
        print('COMPLETE', json.dumps(manifest['compute']), flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc)); save(); raise


if __name__ == '__main__':
    main()
