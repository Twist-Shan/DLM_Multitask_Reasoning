"""Export multi-target pairs from an unmodified, pinned official iGSM generator."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dlm_coupling.tasks.igsm import ancestors, evaluate_graph, operation_count, relation, validate_pair

SHA = 'a1ed1d04600add811beb08b58d9912ed30999642'


def node_id(param): return ','.join(map(str, param))


def serialize(e, root=True):
    if not root and e.param is not None:
        return {'ref': node_id(e.param)}
    if not e.param_list:
        return {'literal': int(e.get_value.a)}
    return {'op': e.op if e.op in ('mul', 'diff') else 'sum',
            'args': [serialize(x, False) for x in e.param_list]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', default=str(ROOT/'data/upstream/igsm'))
    parser.add_argument('--per-relation', type=int, default=2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists(): raise ValueError('Refusing to overwrite dataset')
    upstream = Path(args.upstream).resolve()
    sha = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != SHA: raise ValueError('Wrong upstream revision')
    if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Upstream must remain unmodified')
    sys.path.insert(0, str(upstream))
    from data_gen.pretrain.id_gen import IdGen
    from tools.tools import fix_seed
    fix_seed(args.seed)
    counts = {r: 0 for r in ('independent', 'shared', 'downstream')}
    pairs = []
    for attempt in range(300):
        g = IdGen(max_op=10, max_edge=18, op=6, perm_level=5, detail_level=0)
        g.gen_prob(list(range(23)), p_format='pq')
        p = g.problem
        nodes = {node_id(n): {'name': p.get_ntn(n), 'value': int(p.lookup[n].a),
                             'expression': serialize(p.sketch[n])}
                 for n in p.template.nodes if n[0] != -1}
        values = evaluate_graph(nodes)
        if any(values[k] != v['value'] for k, v in nodes.items()):
            raise ValueError('Expression export mismatch')
        candidates = [n for n in nodes if len(ancestors(nodes, n)) >= 1 and operation_count(nodes[n]['expression']) >= 1]
        chosen = None
        for a in sorted(candidates):
            for b in sorted(candidates):
                if a == b: continue
                r = relation(nodes, a, b)
                if r not in counts or counts[r] >= args.per_relation: continue
                # Distinct answers help detect assignment swaps; predeclared,
                # model-independent filtering, never based on model scores.
                if values[a] == values[b]: continue
                chosen = a, b, r
                break
            if chosen: break
        if not chosen: continue
        a, b, r = chosen
        facts = g.vanilla_prob.strip()
        pair = {'pair_id': f'igsm.s{args.seed}.{attempt:04d}', 'relation': r, 'seed': args.seed,
                'upstream_revision': sha, 'upstream_source_index': attempt,
                'facts': facts, 'facts_sha256': hashlib.sha256(facts.encode()).hexdigest(),
                'modulus': 23, 'nodes': nodes, 'targets': {},
                'original_question': g.ques, 'original_solution': g.sol}
        for label, node in zip(('A', 'B'), (a, b)):
            l, i, j, k = map(int, node.split(','))
            entity = p.N[i+1][k] if l == 0 else p.ln[k]
            closure = ancestors(nodes, node) | {node}
            pair['targets'][label] = {'node': node, 'gold': values[node],
                'question': f'How many {entity} does {p.N[i][j]} have?',
                'ancestors': sorted(closure - {node}),
                'arithmetic_operations': sum(operation_count(nodes[n]['expression']) for n in closure)}
        validate_pair(pair)
        pairs.append(pair); counts[r] += 1
        print(attempt, r, pair['targets'], flush=True)
        if all(v == args.per_relation for v in counts.values()): break
    if not all(v == args.per_relation for v in counts.values()):
        raise ValueError(f'Could not fill strata: {counts}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in pairs), encoding='utf-8')
    print('EXPORTED', output, counts)


if __name__ == '__main__': main()
