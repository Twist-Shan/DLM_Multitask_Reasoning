"""Pair-level comparisons and offline audit of late-answer interventions."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dlm_coupling.tasks.igsm import evaluate_graph
from dlm_coupling.tasks.igsm_explicit import parse_answers


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def interval(values):
    rng = random.Random(123)
    n = len(values)
    samples = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(10000))
    return [samples[249], samples[9749]]


def evaluate_with_override(pair, source, target, value):
    """Post-hoc graph prediction; this is never supplied to model generation."""
    nodes = dict(pair['nodes'])
    nodes[pair['targets'][source]['node']] = {'expression': {'literal': value}}
    return evaluate_graph(nodes, modulus=None)[pair['targets'][target]['node']]


def trailing_json_answers(text, labels):
    """Exploratory format diagnosis only; never replaces frozen primary scoring."""
    if text.count('<think>') != text.count('</think>'):
        return {}, 'unfinished_reasoning'
    text = text.rsplit('</think>', 1)[-1].strip()
    if text.endswith('```'):
        text = text[:-3].rstrip()
    for i in reversed([i for i, c in enumerate(text) if c == '{']):
        answers, error = parse_answers(text[i:], labels)
        if error is None:
            return answers, None
    return {}, 'no_valid_trailing_json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--models', nargs='+', choices=['dream', 'illada'], default=['dream', 'illada'])
    args = ap.parse_args()
    assert not args.out.exists(), 'Use a new output file'
    result = {'seed': 123, 'bootstrap_unit': 'pair', 'models': {}}
    root = args.artifacts
    source_hash = hashlib.sha256(b''.join(p.read_bytes() for p in sorted((root/'src').rglob('*.py')))).hexdigest()
    for family in args.models:
        base = root/'runs'/f'screen30-{family}'
        base_manifest = json.loads((base/'manifest.json').read_text())
        pairs = {p['pair_id']: p for p in read_rows(root/base_manifest['args']['data'])}
        rows = read_rows(base/'results.jsonl')
        bykey = {(r['pair_id'], r['order']): r for r in rows}
        model = {'comparisons': [], 'events': {}, 'interventions': [], 'format_diagnostics': []}
        for row in rows:
            if row['parse_error']:
                answers, error = trailing_json_answers(row['trace']['text'], row['order'])
                model['format_diagnostics'].append({'pair_id': row['pair_id'], 'relation': row['relation'], 'order': row['order'],
                    'primary_error': row['parse_error'], 'eos_found': row['trace']['eos_found'],
                    'trailing_json_answers': answers, 'trailing_json_error': error,
                    'trailing_json_correct': {q: answers.get(q) == row['gold'][q] for q in row['order']}})
        for relation in ('all', 'independent', 'shared', 'downstream'):
            ids = sorted({r['pair_id'] for r in rows if relation == 'all' or r['relation'] == relation})
            for order in ('AB', 'BA'):
                differences = []
                for pid in ids:
                    single = sum(sum(bykey[pid, q]['correct'].values()) for q in ('A', 'B')) / 2
                    joint = sum(bykey[pid, order]['correct'].values()) / 2
                    differences.append(joint-single)
                model['comparisons'].append({'relation': relation, 'order': order, 'pairs': len(ids),
                    'delta_accuracy': sum(differences)/len(ids), 'percentile_ci95': interval(differences),
                    'improved_pairs': sum(v > 0 for v in differences), 'worsened_pairs': sum(v < 0 for v in differences)})
        folder = root/'interp'/family
        manifest = json.loads((folder/'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        assert manifest['seed'] == 123 and manifest['source_digest'] == source_hash
        assert manifest['base_results_sha256'] == hashlib.sha256((base/'results.jsonl').read_bytes()).hexdigest()
        events = read_rows(folder/'events.jsonl')
        assert len(events) == manifest['events'] == 60
        assert Counter((e['pair_id'], e['order']) for e in events) == Counter(k for k in bykey if len(k[1]) == 2)
        for ev in events:
            row = bykey[ev['pair_id'], ev['order']]
            steps = {w['output_position']: s['step'] for s in row['trace']['steps'] for w in s['writes']}
            for label, span in ev['answers'].items():
                assert span['value'] == row['answers'][label]
                assert span['token_ids'] == [row['trace']['output_token_ids'][p] for p in span['positions']]
                assert span['start_step'] == min(steps[p] for p in span['positions'])
                assert span['complete_step'] == max(steps[p] for p in span['positions'])
                assert span['blocks'] == sorted({p//row['trace']['block_size'] for p in span['positions']})
        for relation, order in [(r, o) for r in ('independent', 'shared', 'downstream') for o in ('all', 'AB', 'BA')]:
            counts = Counter()
            for ev in events:
                if ev['relation'] != relation or (order != 'all' and ev['order'] != order):
                    continue
                counts['rows'] += 1
                if ev['error']:
                    counts['unmapped'] += 1
                    continue
                a, b = ev['answers']['A'], ev['answers']['B']
                same = a['blocks'] == b['blocks'] and len(a['blocks']) == 1
                counts['same_block' if same else 'different_blocks'] += 1
                direction = 'A_before_B' if a['complete_step'] < b['start_step'] else 'B_before_A' if b['complete_step'] < a['start_step'] else 'interleaved'
                counts[('same_block_' if same else 'different_blocks_') + direction] += 1
                counts['eligible_rows'] += bool(ev['eligible_directions'])
                layout = 'AB' if min(a['positions']) < min(b['positions']) else 'BA'
                counts['output_matches_requested_order'] += layout == ev['order']
            model['events'][relation if order == 'all' else relation+'_'+order] = dict(counts)
        forks = read_rows(folder/'forks.jsonl')
        selected = []; selected_ids = set(); counts = Counter()
        for ev in events:
            if ev['eligible_directions'] and ev['pair_id'] not in selected_ids and counts[ev['relation']] < manifest['args']['per_relation']:
                selected.append((ev['pair_id'], ev['order']))
                selected_ids.add(ev['pair_id']); counts[ev['relation']] += 1
        assert selected == [(f['pair_id'], f['order']) for f in forks]
        assert dict(Counter(f['relation'] for f in forks)) == manifest['forks_by_relation']
        assert len({f['pair_id'] for f in forks}) == len(forks)
        nfe = 0
        for record in forks:
            row = bykey[record['pair_id'], record['order']]
            f = record['fork']
            assert record['control_exact'] and record['control']['output_token_ids'] == row['trace']['output_token_ids']
            initial = {w['output_position']: w['token_id'] for s in row['trace']['steps'] if s['step'] <= f['cut_step'] for w in s['writes']}
            assert [initial[p] for p in f['positions']] == f['old_token_ids']
            assert all(p not in initial for p in f['target_positions'])
            assert len({p//row['trace']['block_size'] for p in f['positions']+f['target_positions']}) == 1
            assert len(str(f['old_value'])) == len(str(f['new_value'])) and abs(f['old_value']-f['new_value']) == 1
            assert len(f['positions']) == len(f['new_token_ids'])
            for branch in ('control', 'changed'):
                state = dict(initial)
                if branch == 'changed':
                    state.update(zip(f['positions'], f['new_token_ids']))
                output = record[branch]
                assert parse_answers(output['text'], row['order']) == (output['answers'], output['parse_error'])
                assert output['nfe'] == len(output['writes'])
                for write in output['writes']:
                    assert write['output_position'] not in state
                    state[write['output_position']] = write['token_id']
                assert set(state) == set(range(len(output['output_token_ids'])))
                assert [state[p] for p in range(len(state))] == output['output_token_ids']
                nfe += output['nfe']
            nfe += 2
            for probe in record['probes']:
                assert all(0 <= p <= 1 for p in probe['baseline_digit_probabilities'])
                assert math.isclose(probe['sum_logprob'], sum(math.log(max(p, 1e-45)) for p in probe['baseline_digit_probabilities']), abs_tol=1e-12)
            delta = record['probes'][1]['sum_logprob']-record['probes'][0]['sum_logprob']
            assert math.isclose(delta, record['delta_sum_logprob'])
            changed = record['changed']
            target_changed = None if changed['parse_error'] else changed['answers'][f['target']] != row['answers'][f['target']]
            assert target_changed == record['target_changed']
            model['interventions'].append({'pair_id': record['pair_id'], 'relation': record['relation'], 'order': record['order'],
                'direction': f['source']+'->'+f['target'], 'old_source': f['old_value'], 'new_source': f['new_value'],
                'baseline_answers': row['answers'], 'changed_answers': changed['answers'], 'gold': row['gold'],
                'source_value_preserved': None if changed['parse_error'] else changed['answers'][f['source']] == f['new_value'],
                'graph_target_with_old_source': evaluate_with_override(pairs[row['pair_id']], f['source'], f['target'], f['old_value']),
                'graph_target_with_new_source': evaluate_with_override(pairs[row['pair_id']], f['source'], f['target'], f['new_value']),
                'target_changed': target_changed, 'parse_error': changed['parse_error'], 'delta_sum_logprob': delta})
        assert nfe == manifest['experimental_nfe']
        model['intervention_nfe'] = nfe
        result['models'][family] = model
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('INTERVENTION AUDIT PASSED', args.out)


if __name__ == '__main__':
    main()
