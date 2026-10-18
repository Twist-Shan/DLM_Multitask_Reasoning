import ast
from collections import Counter
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dlm_coupling import answer_graphs as task


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) + 10 for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(chr(i - 10) for i in ids)
    def apply_chat_template(self, messages, **kwargs):
        prefix = '<user>' + messages[0]['content'] + '</user><assistant>\n'
        return prefix if len(messages) == 1 else prefix + messages[1]['content'] + '</assistant>'


class Backend:
    tokenizer = Tokenizer()
    mask_id = 5
    def predict(self, ids, positions, probes):
        # Deliberately choose an out-of-domain token, with position-varying confidence.
        return [dict(token_id=ord('?') + 10, confidence=float(p), probabilities={str(x): 0.01 for x in ps})
                for p, ps in zip(positions, probes)]


class MergedTokenizer(Tokenizer):
    def __init__(self, prefix=' '):
        self.merges = {prefix + s: 1000 + i for i, s in enumerate(task.SYMBOLS)}
    def encode(self, text, **kwargs):
        ids = []
        while text:
            match = next((s for s in self.merges if text.startswith(s)), None)
            if match:
                ids.append(self.merges[match]); text = text[len(match):]
            else:
                ids.append(ord(text[0]) + 10); text = text[1:]
        return ids
    def decode(self, ids, **kwargs):
        inverse = {v: k for k, v in self.merges.items()}
        return ''.join(inverse[i] if i in inverse else chr(i - 10) for i in ids)


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = task.dataset(per_cell=2)

    def test_reproducibility_and_no_input_family_overlap(self):
        self.assertEqual(self.data, task.dataset(per_cell=2))
        counts = Counter((c['family'], c['graph']) for c in self.data['cases'])
        self.assertEqual(set(counts.values()), {2})
        seen = set()
        for case in self.data['cases']:
            keys = {task.prompt(case)} | {task.prompt(p['Source']) for p in task.source_pairs(case)}
            self.assertFalse(keys & seen)
            seen.update(keys)
            self.assertEqual(task.digest(task.prompt(case)), case['input_sha256'])

    def test_all_source_controls_and_do_oracle(self):
        for case in self.data['cases']:
            self.assertTrue(task.validate(case))
            self.assertEqual(task.source(case, 'A', 'self'), case)
            gold = task.evaluate(case)
            independent = case['nodes'][-1]['name']
            for pair in task.source_pairs(case):
                if pair['kind'] == 'change':
                    self.assertEqual(pair['Source_values'], task.evaluate(case, {pair['site']: pair['Source_values'][pair['site']]}))
                    if pair['site'] != independent:
                        self.assertEqual(pair['Source_values'][independent], gold[independent])
                else:
                    self.assertEqual(pair['Source_values'], gold)

    def test_printed_numeric_equations_have_the_oracle_answers(self):
        for case in self.data['cases']:
            if case['family'] not in ('addition', 'subtraction'):
                continue
            expressions = {line.split(': ')[0][-1]: line.split(' = ')[1][:-1]
                           for line in task.prompt(case).splitlines() if line.startswith('Definition of ')}
            values = {}
            for node in case['nodes']:
                # Only locally generated arithmetic expressions; check the AST allowlist first.
                tree = ast.parse(expressions[node['name']], mode='eval')
                self.assertTrue(all(isinstance(n, (ast.Expression, ast.BinOp, ast.Add, ast.Sub, ast.Constant, ast.Name, ast.Load)) for n in ast.walk(tree)))
                values[node['name']] = eval(compile(tree, '<test>', 'eval'), {'__builtins__': {}}, values)
            self.assertEqual({k: str(v) for k, v in values.items()}, task.evaluate(case))

    def test_subtraction_reverses_the_edge_effect(self):
        case = next(c for c in self.data['cases'] if c['family'] == 'subtraction' and c['graph'] == 'chain')
        a, b, c, d, independent = [n['name'] for n in case['nodes']]
        before = task.evaluate(case); after = task.evaluate(task.source(case, a, 'change', 1))
        self.assertEqual([int(after[k]) - int(before[k]) for k in (a, b, c, d, independent)], [1, -1, 1, -1, 0])

    def test_boolean_reconvergent_cancellation_is_not_no_dependency(self):
        case = task.make_case('boolean', 'diamond', task.stream(123, 'cancellation'), list(task.FIELDS), list(task.FIELDS))
        for node in case['nodes']:
            node.update(op='XOR', params=[0, 0])
        before = task.evaluate(case); after = task.evaluate(task.source(case, 'A'))
        self.assertIn('D', task.descendants(case, 'A'))
        self.assertEqual(before['D'], after['D'])
        self.assertNotEqual(before['B'], after['B'])
        self.assertNotEqual(before['C'], after['C'])

    def test_lookup_same_value_control_matches_lexical_edit(self):
        for case in self.data['cases']:
            if case['family'] != 'lookup':
                continue
            for site in task.FIELDS:
                original = next(n for n in case['nodes'] if n['name'] == site)['table']
                for value in task.alternatives(case, site):
                    edits = []
                    for kind in ('change', 'same_value'):
                        changed = next(n for n in task.source(case, site, kind, value)['nodes'] if n['name'] == site)['table']
                        edits.append([(original[k], v) for k, v in changed.items() if v != original[k]])
                    self.assertEqual(edits[0], edits[1])
                    self.assertEqual(len(edits[0]), 1)

    def test_canvas_no_gold_leakage_and_full_vocabulary_decoder(self):
        backend = Backend(); builder = task.CanvasBuilder(backend)
        for case in self.data['cases'][::2]:
            for layout in task.layouts(case).values():
                cc = builder.build(case, task.FIELDS, layout)
                positions = [p for ps in cc['spans'].values() for p in ps]
                self.assertTrue(all(cc['ids'][p] == backend.mask_id for p in positions))
                self.assertEqual(cc['unused'], [])
                outputs = {policy: task.decode(backend, cc, policy) for policy in ('confidence', 'fixed_layout')}
                self.assertEqual(outputs['fixed_layout']['writes'][0]['position'], positions[0])
                self.assertEqual(outputs['confidence']['writes'][0]['position'], positions[-1])
                self.assertTrue(all(v is None for v in outputs['confidence']['values'].values()))
                self.assertEqual(outputs['confidence']['logical_forwards'], len(positions))

    def test_cyclic_or_mismatched_graph_rejected(self):
        from copy import deepcopy
        case = deepcopy(self.data['cases'][0])
        case['nodes'][0]['parents'] = [case['nodes'][-1]['name']]
        with self.assertRaises(ValueError):
            task.validate(case)

    def test_space_symbol_merge_allowed_but_label_merge_rejected(self):
        case = next(c for c in self.data['cases'] if c['family'] == 'lookup')
        backend = Backend(); backend.tokenizer = MergedTokenizer()
        cc = task.CanvasBuilder(backend).build(case, task.FIELDS, 'ABCDE')
        for symbol in task.SYMBOLS:
            ids = cc['ids'].copy()
            for name, positions in cc['spans'].items():
                self.assertEqual(len(positions), 1)
                ids[positions[0]] = cc['maps'][name][symbol][0]
            self.assertEqual(task.read_values(cc, ids), dict.fromkeys(task.FIELDS, symbol))
            self.assertTrue(backend.tokenizer.decode(ids).endswith('; '.join(f'{k} = {symbol}' for k in task.FIELDS) + '.</assistant>'))
        backend.tokenizer = MergedTokenizer(' = ')
        with self.assertRaisesRegex(ValueError, 'non-whitespace'):
            task.CanvasBuilder(backend).build(case, task.FIELDS, 'ABCDE')

    def test_saved_behavior_replay_rejects_wrong_gold_and_schedule(self):
        from copy import deepcopy
        from analyze_answer_graphs import verify_row
        case = next(c for c in self.data['cases'] if c['family'] == 'boolean')
        backend = Backend(); cc = task.CanvasBuilder(backend).build(case, task.FIELDS, task.layouts(case)['reverse'])
        positions = [p for ps in cc['spans'].values() for p in ps]
        initial = task.predict(backend, cc, cc['ids'], positions)
        initial_ids = cc['ids'].copy()
        for p, prediction in zip(positions, initial):
            initial_ids[p] = prediction['token_id']
        row = dict(expected=task.evaluate(case), requested=task.FIELDS, layout_name='reverse',
                   canvas={k: v for k, v in cc.items() if k != 'maps'}, value_token_ids=cc['maps'],
                   t0=task.read_values(cc, initial_ids), t0_predictions=initial,
                   **task.decode(backend, cc, 'fixed_layout', initial))
        verify_row(row, case)
        wrong = deepcopy(row); wrong['expected']['A'] = 'invalid'
        with self.assertRaises(AssertionError):
            verify_row(wrong, case)
        wrong = deepcopy(row); wrong['writes'][0]['position'] = positions[-1]
        with self.assertRaises(AssertionError):
            verify_row(wrong, case)


if __name__ == '__main__':
    unittest.main()
