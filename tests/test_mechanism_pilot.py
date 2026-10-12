import unittest
from dlm_coupling.mechanism_pilot import dataset,choose,HeadPatch
from dlm_coupling.baseline_repair import splits
from dlm_coupling.shared_followup import branch_answers,cases
from dlm_coupling.shared_causal import examples

class MechanismTests(unittest.TestCase):
    def test_disjoint_and_selective(self):
        data=dataset();self.assertEqual(len(data),10)
        old={tuple(sorted(p.items())) for c in examples(6,123) for p in (c['base'],c['donor'],c['same'])}
        old.update(tuple(sorted(c[v].items())) for c in cases(10,124) for v in ('base','donor','same'))
        old.update(tuple(sorted(c['base'].items())) for rr in splits().values() for c in rr)
        keys=[tuple(sorted(p.items())) for c in data for p in c['variants'].values()]
        self.assertEqual(len(set(keys)),50);self.assertFalse(set(keys)&old)
        for c in data:
            g={v:branch_answers(p) for v,p in c['variants'].items()}
            self.assertEqual(g['base'],g['same'])
            for v,da,db in [('C',1,2),('U',1,0),('V',0,1)]:
                self.assertEqual((g[v]['A']-g['base']['A'],g[v]['B']-g['base']['B']),(da,db))
    def test_head_slice(self):
        try:import torch
        except (ImportError,PermissionError) as e:self.skipTest(str(e))
        m=torch.nn.Identity();x=torch.arange(24.).reshape(1,3,8)
        with HeadPatch(m,[1],2,4) as cap:m(x)
        self.assertTrue(torch.equal(cap.captured,x[:,[1],4:6]))
        with HeadPatch(m,[1],2,4,torch.zeros_like(cap.captured)):y=m(x)
        expected=x.clone();expected[:,[1],4:6]=0
        self.assertTrue(torch.equal(y,expected));self.assertTrue(torch.equal(m(x),x))
    def test_selection_requires_eligible(self):
        rr=[dict(kind='grid',config=n,component='attention',layer=l,induced=True,eligible=e,correct=True) for n,l,e in [('a',7,False),('b',15,True)]]
        self.assertEqual(choose(rr),'b')
