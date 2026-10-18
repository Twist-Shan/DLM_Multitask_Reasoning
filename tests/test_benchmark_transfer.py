import unittest
from dlm_coupling import benchmark_transfer as t


class TransferTests(unittest.TestCase):
    def test_alias_and_multiset(self):
        self.assertEqual(t.answer_score('The Denver, Colorado', ['Denver', 'Denver Colorado']), {'em': 1., 'f1': 1.})
        self.assertAlmostEqual(t.answer_score('red red blue', ['red blue'])['f1'], .8)

    def test_strict_format(self):
        self.assertIsNone(t.parse_answers('A: x\nA: y\nB: z', 'AB'))
        self.assertIsNone(t.parse_answers('B: x\nexplanation', 'B'))
        self.assertEqual(t.parse_answers('B: Denver\nA: Colorado', 'BA')['A'], 'Colorado')

    def test_coordinate_direction_and_label_check(self):
        import re
        mapping = [(re.compile(r'([A-Z]) right ([A-Z])'), r'\1_right_\2'),
                   (re.compile(r'([A-Z]) above ([A-Z])'), r'\1_over_\2'),
                   (re.compile(r'query ([A-Z]) ([A-Z])'), r'\1_query_\2')]
        row = dict(story=['X right Y', 'Y above Z'], question='query X Z', label='upper-right', k_hop='2')
        audit = t.spatial_audit(row, mapping)
        self.assertEqual(audit['path'], ['X', 'Y', 'Z'])
        self.assertEqual(audit['A_answer'], 'right')
        with self.assertRaisesRegex(ValueError, 'published_label_mismatch'):
            t.spatial_audit(dict(row, label='lower-left'), mapping)
        with self.assertRaisesRegex(ValueError, 'inconsistent_coordinates'):
            t.spatial_audit(dict(row, story=row['story'] + ['Z right X']), mapping)

    def test_root_ancestry_no_gold_injection(self):
        row = dict(id='3hop2__test', answerable=True, question='Main question?', answer='secret-final', answer_aliases=[],
                   question_decomposition=[dict(question='Entity >> birthplace', answer='secret-root'),
                                           dict(question='Who follows #1?', answer='secret-middle'),
                                           dict(question='Where is #2?', answer='secret-final')])
        case = t.musique_case(row)
        self.assertEqual(case['dependency_edges'], [(0, 1), (1, 2)])
        self.assertNotIn('secret', case['questions']['A'])

    def test_completed_final_diagnostic_does_not_mine_reasoning(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from analyze_benchmark_transfer import completed_final
        self.assertIsNone(completed_final('<think>draft B: correct but unfinished'))
        self.assertEqual(completed_final('<think>draft</think>\nB: answer'), 'B: answer')
        self.assertIsNone(completed_final('<think>a<think>b</think>B: answer'))


if __name__ == '__main__':
    unittest.main()
