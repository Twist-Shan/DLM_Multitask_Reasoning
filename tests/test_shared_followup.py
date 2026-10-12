import unittest
from types import SimpleNamespace
from unittest.mock import patch
from dlm_coupling.shared_causal import examples
from dlm_coupling.shared_followup import cases, gold, aligned_donor, rollout


class FollowupTests(unittest.TestCase):
    def test_new_cases_and_selective_counterfactuals(self):
        old={tuple(sorted(p.items())) for r in examples() for p in (r['base'],r['donor'],r['same'])}
        rows=cases()
        self.assertEqual(len(rows),20)
        self.assertEqual(rows,cases(seed=124))
        for r in rows:
            if r['suite']=='replication':
                for k in ('base','donor','same'): self.assertNotIn(tuple(sorted(r[k].items())),old)
            else:
                base=gold(r)
                for k in 'CUV': self.assertEqual(base[k],gold(r,'same')[k])
                self.assertEqual(gold(r,site='U')['B'],base['B'])
                self.assertEqual(gold(r,site='V')['A'],base['A'])
                self.assertEqual(gold(r,site='C')['A'],base['A']+1)
                self.assertEqual(gold(r,site='C')['B'],base['B']+2)
                self.assertEqual(gold(r,site='U')['A'],base['A']+2)
                self.assertEqual(gold(r,site='V')['B'],base['B']+3)
                for k in ('base','donor','same'):
                    self.assertTrue(all(0<v<10 for v in r[k].values()))

    def test_donor_alignment_copies_only_partial_answers(self):
        donor=[9,2,5,5,8];recipient=[1,2,7,5,8]
        self.assertEqual(aligned_donor(donor,recipient,[2,3],5),[9,2,7,5,8])
        self.assertEqual(donor,[9,2,5,5,8])
        with self.assertRaises(ValueError): aligned_donor([5,2,5,5,8],recipient,[2,3],5)
        with self.assertRaises(ValueError): aligned_donor([1],recipient,[2,3],5)

    def test_sustained_recaptures_at_every_step_and_single_does_not(self):
        class FakePatch:
            def __init__(self,*args): self.captured='captured'
            def __enter__(self): return self
            def __exit__(self,*args): pass
        b=SimpleNamespace(mask_id=5)
        b.predict=lambda ids,ps,probes:[dict(token_id=10+p,confidence=.9,token_text='1') for p in ps]
        with patch('dlm_coupling.shared_followup.ResidualPatch',FakePatch):
            for mode,count,nfe in [('single',1,5),('sustained',4,8)]:
                end,writes,n=rollout(b,[1,5,5,5,5],dict(A=[1,2],B=[3,4]),object(),[9,5,5,5,5],[0],mode)
                self.assertEqual(n,nfe)
                self.assertEqual(sum(w['patched'] for w in writes),count)
                for w in writes:
                    if w['patched']:
                        self.assertEqual(w['donor_state'][1:],w['recipient_state'][1:])
                        self.assertEqual(w['donor_state'][0],9)
                self.assertEqual(end,[1,11,12,13,14])


if __name__=='__main__': unittest.main()
