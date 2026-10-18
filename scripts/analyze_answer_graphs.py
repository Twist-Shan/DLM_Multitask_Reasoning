"""Replay saved graph calibration and summarize paired, case-level effects."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dlm_coupling import answer_graphs as task


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def mean(values):
    return sum(values) / len(values)


def interval(values, seed=123, draws=4000):
    rng = random.Random(seed)
    samples = sorted(mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(draws))
    return dict(mean=mean(values), ci95=[samples[int(draws * .025)], samples[int(draws * .975)]], groups=len(values))


def verify_row(row, case):
    expected = task.evaluate(case)
    cc = row['canvas']
    if row['expected'] != {k: expected[k] for k in row['requested']}:
        raise AssertionError('Gold mismatch')
    if cc['prompt'] != task.prompt(case, row['requested']) or cc['unused']:
        raise AssertionError('Prompt/tail mismatch')
    if row['layout_name'] == 'single':
        wanted_layout = row['requested']
        if len(wanted_layout) != 1 or row['policy'] != 'confidence':
            raise AssertionError('Invalid single condition')
    else:
        wanted_layout = task.layouts(case)[row['layout_name']]
        if row['requested'] != task.FIELDS or row['policy'] not in ('confidence', 'fixed_layout'):
            raise AssertionError('Invalid joint condition')
    if cc['layout'] != wanted_layout or list(cc['spans']) != list(wanted_layout):
        raise AssertionError('Wrong field order')
    maps = row['value_token_ids']
    positions = [p for ps in cc['spans'].values() for p in ps]
    if len(set(positions)) != len(positions) or any(set(maps[k]) != set(task.domain(case['family'])) for k in cc['spans']):
        raise AssertionError('Invalid slot or value map')
    def values(ids):
        return task.read_values(dict(cc, maps=maps), ids)
    ids = cc['ids'].copy()
    mask = ids[positions[0]]
    if any(ids[p] != mask for p in positions) or len(row['t0_predictions']) != len(positions):
        raise AssertionError('Initial answers not all masked')
    t0_ids = ids.copy()
    for p, pred in zip(positions, row['t0_predictions']):
        t0_ids[p] = pred['token_id']
    if values(t0_ids) != row['t0']:
        raise AssertionError('t0 readout mismatch')
    for step, write in enumerate(row['writes']):
        remaining = [p for p in positions if ids[p] == mask]
        if write['remaining'] != remaining or len(write['predictions']) != len(remaining):
            raise AssertionError('Eligible set mismatch')
        j = max(range(len(remaining)), key=lambda i: (write['predictions'][i]['confidence'], -remaining[i])) if row['policy'] == 'confidence' else 0
        if write['position'] != remaining[j] or write['token_id'] != write['predictions'][j]['token_id'] or write['token_id'] == mask:
            raise AssertionError('Commitment policy mismatch')
        if step == 0 and write['predictions'] != row['t0_predictions']:
            raise AssertionError('Initial forward reuse mismatch')
        ids[write['position']] = write['token_id']
    if len(row['writes']) != len(positions) or row['ids'] != ids or row['values'] != values(ids):
        raise AssertionError('Incomplete or inconsistent decode')
    if row['logical_forwards'] != len(positions) or row['new_forwards'] != len(positions) - 1:
        raise AssertionError('Forward accounting mismatch')


def analyze(run):
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    data = json.loads((run / 'dataset.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or manifest['dataset_sha256'] != task.digest(data):
        raise AssertionError('Run incomplete or dataset changed')
    cases = {c['id']: c for c in data['cases']}
    rows = read_rows(run / 'behavior.jsonl')
    seen, keyed = set(), {}
    aggregate = defaultdict(list)
    forwards = 0
    t0_seen = set()
    for row in rows:
        case = cases[row['case']]
        verify_row(row, case)
        if (row['family'], row['graph']) != (case['family'], case['graph']):
            raise AssertionError('Wrong case family')
        identity = (row['case'], row['layout_name'], row['requested'], row['policy'])
        if identity in seen:
            raise AssertionError('Duplicate output')
        seen.add(identity); keyed[identity] = row
        t0_identity = identity[:3]
        forwards += row['new_forwards'] + int(t0_identity not in t0_seen)
        t0_seen.add(t0_identity)
        request = row['requested'] if row['layout_name'] == 'single' else 'joint'
        aggregate[(case['family'], case['graph'], row['layout_name'], request, row['policy'])].append(row)
    expected = {(cid, 'single', k, 'confidence') for cid in cases for k in task.FIELDS}
    expected |= {(cid, layout, task.FIELDS, policy) for cid in cases for layout in ('topological', 'reverse') for policy in ('confidence', 'fixed_layout')}
    if seen != expected or len(rows) != manifest['rows'] or forwards != manifest['compute']['forwards']:
        raise AssertionError('Incomplete grid or forward budget mismatch')
    cells = []
    for key, group in sorted(aggregate.items()):
        item = dict(zip(('family', 'graph', 'layout', 'requested', 'policy'), key))
        item['groups'] = len(group)
        for event in ('t0', 'values'):
            item[event] = dict(
                field_accuracy=mean([mean([r[event][k] == v for k, v in r['expected'].items()]) for r in group]),
                all_correct=sum(all(r[event][k] == v for k, v in r['expected'].items()) for r in group),
                invalid_fields=sum(v is None for r in group for v in r[event].values()),
            )
        cells.append(item)
    contrasts = []
    for family in data['families']:
        for graph in data['graphs']:
            selected = [c for c in cases.values() if (c['family'], c['graph']) == (family, graph)]
            for event in ('t0', 'values'):
                for policy in ('confidence', 'fixed_layout'):
                    differences = []
                    for case in selected:
                        cid = case['id']; gold = task.evaluate(case)
                        before = keyed[(cid, 'topological', task.FIELDS, policy)][event]
                        after = keyed[(cid, 'reverse', task.FIELDS, policy)][event]
                        differences.append(mean([(after[k] == gold[k]) - (before[k] == gold[k]) for k in task.FIELDS]))
                    contrasts.append(dict(family=family, graph=graph, event=event, policy=policy,
                                          contrast='reverse_minus_topological_mean_field_accuracy', **interval(differences)))
            for layout in ('topological', 'reverse'):
                for event in ('t0', 'values'):
                    differences = []
                    for case in selected:
                        cid = case['id']; gold = task.evaluate(case)
                        joint = keyed[(cid, layout, task.FIELDS, 'confidence')][event]
                        differences.append(mean([(joint[k] == gold[k]) - (keyed[(cid, 'single', k, 'confidence')][event][k] == gold[k]) for k in task.FIELDS]))
                    contrasts.append(dict(family=family, graph=graph, event=event, policy='confidence', layout=layout,
                                          contrast='joint_minus_single_mean_field_accuracy', **interval(differences)))
    return dict(status='verified', protocol=task.PROTOCOL, seed=data['seed'], cases=len(cases), rows=len(rows),
                model=manifest['model'], compute=manifest['compute'], tokenizer_alignments=manifest['tokenizer_alignments'],
                cells=cells, contrasts=contrasts, inference='Exploratory calibration only; intervals resample graph instances, not fields; no mechanistic result.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); result = analyze(args.run)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ('status', 'cases', 'rows', 'compute')}))


if __name__ == '__main__':
    main()
