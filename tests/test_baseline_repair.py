import unittest
from dlm_coupling.baseline_repair import splits,natural_values,choose
from dlm_coupling.shared_causal import examples
from dlm_coupling.shared_followup import cases

class RepairTests(unittest.TestCase):
    def test_no_overlap(self):
        old={tuple(sorted(p.items())) for c in examples(6,123) for p in (c['base'],c['donor'],c['same'])}
        old.update(tuple(sorted(c[v].items())) for c in cases(10,124) for v in ('base','donor','same'))
        d=splits();new=[tuple(sorted(c['base'].items())) for rows in d.values() for c in rows]
        self.assertEqual(len(new),28);self.assertEqual(len(set(new)),28);self.assertFalse(old&set(new));self.assertEqual(d,splits())
    def test_parser_rejects_ambiguous(self):
        self.assertEqual(natural_values('work\nFINAL: A = 23; B = 35.'),dict(A=23,B=35))
        for text in ('A = 23; B = 35','FINAL: A = x; B = 35','FINAL: A = 23; B = 35\nFINAL: A = 23; B = 35'):
            self.assertEqual(natural_values(text),dict(A=None,B=None))
        self.assertEqual(natural_values('<think>FINAL: A = 99; B = 99</think>FINAL: A = 23; B = 35'),dict(A=23,B=35))
        for text in ('<think>\nFINAL: A = 23; B = 35','FINAL: A = 23; B = 35.5','Mention FINAL: A = 23; B = 35'):
            self.assertEqual(natural_values(text),dict(A=None,B=None))
    def test_selection_excludes_control(self):
        rows=[dict(suite='branches',variant=v,correct=c,nfe=n) for v,c,n in [('old_masked',True,1),('answer_only',True,4),('generated_intermediates',True,10),('natural',False,768)]]
        self.assertEqual(choose(rows,'branches'),'answer_only')
