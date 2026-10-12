import unittest
from dlm_coupling.headbank_pilot import validation_cases,BankPatch
from dlm_coupling.mechanism_pilot import dataset

class BankTests(unittest.TestCase):
    def test_new_cases(self):
        rr=validation_cases();self.assertEqual(len(rr),6)
        keys=[tuple(sorted(p.items())) for c in rr for p in c['variants'].values()]
        old={tuple(sorted(p.items())) for c in dataset() for p in c['variants'].values()}
        self.assertEqual(len(set(keys)),30);self.assertFalse(set(keys)&old)
    def test_multiple_head_slice(self):
        try:import torch
        except (ImportError,PermissionError) as e:self.skipTest(str(e))
        x=torch.arange(3*4096.).reshape(1,3,4096);mod=torch.nn.Identity()
        with BankPatch(mod,[1],[0,31]) as cap:mod(x)
        self.assertEqual(tuple(cap.captured.shape),(1,1,256))
        with BankPatch(mod,[1],[0,31],torch.zeros_like(cap.captured)):y=mod(x)
        expected=x.clone();expected[:,1,:128]=0;expected[:,1,-128:]=0
        self.assertTrue(torch.equal(y,expected));self.assertTrue(torch.equal(mod(x),x))
