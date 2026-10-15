import unittest
from types import SimpleNamespace

from dlm_coupling import precommit as v1, precommit_v2 as v2, precommit_v21 as v21, precommit_v22 as v22
from test_precommit_v2 import Fake
from test_precommit import torch


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=v2.dataset();cls.groups=cls.data['groups'][:30]

    def test_compensation_both_signs_and_independent_candidate(self):
        for g in self.data['groups']:
            p=g['p'];base=v2.gold(p,'downstream')
            for sign in (-1,1):
                q=v22.source(p,'A_change_B_fixed',sign)
                self.assertEqual(v2.gold(q,'downstream'),dict(A=base['A']+sign,B=base['B']))
                q=v22.source(p,'A_fixed_B_change',sign)
                self.assertEqual(v2.gold(q,'downstream'),dict(A=base['A'],B=base['B']+sign))
                q=v22.source(p,'change_A',sign)
                self.assertEqual(v2.gold(q,'independent')['B'],v2.gold(p,'independent')['B'])
        with self.assertRaises(ValueError):v22.source(p,'change_A',0)

    def test_disjoint_replication_and_explicit_reserved_family_collision(self):
        audit=v22.audit_splits(self.data)
        self.assertEqual(audit['intersections']['calibration/discovery'],0)
        self.assertEqual(audit['confirmation_target_collisions'],[])
        self.assertEqual(len(audit['confirmation_family_collisions']),4)

    def test_canvas_and_schedule(self):
        b=Fake();builder=v22.CanvasBuilder(b);p=self.groups[0]['p']
        old=v21.CanvasBuilder(b).build(p);cc=builder.build(p,'downstream')
        for k in old:self.assertEqual(cc[k],old[k])
        for rel in ('downstream','independent'):
            cc=builder.build(p,rel)
            self.assertEqual(b.tokenizer.decode([cc['ids'][i] for i in cc['prompt_A']]),cc['prompt_A_text'])
            for k in v22.SOURCE_KINDS:
                for sign in (-1,1):v21.check_alignment(cc,builder.build(v22.source(p,k,sign),rel))
        self.assertEqual(len(v22.schedule('downstream','change_A')),26)
        self.assertEqual(len(v22.schedule('downstream','A_change_B_fixed')),14)
        self.assertEqual(len(v22.schedule('independent','change_A')),4)
        self.assertEqual(sum(len(v22.schedule(r,k)) for r,ks in v22.specification()['relation_sources'].items() for k in ks)*60+360,7440)

    def test_resample_derangements_reproducible_and_no_value_filter(self):
        pairs=v22.resample_pairs(self.groups)
        self.assertEqual(pairs,v22.resample_pairs(self.groups))
        self.assertEqual(len(pairs),150)
        for rep in range(5):
            rr=[r for r in pairs if r['repetition']==rep]
            self.assertEqual({r['Target'] for r in rr},{r['Source'] for r in rr})
            self.assertTrue(all(r['Target']!=r['Source'] for r in rr))
        with self.assertRaises(ValueError):v22.resample_pairs(self.groups[:1])

    @unittest.skipIf(torch is None,'optional CPU torch hook test')
    def test_sustained_uses_each_source_layer_and_cleans_hooks(self):
        class Block(torch.nn.Module):
            def __init__(self,scale):super().__init__();self.scale=scale
            def forward(self,h):return h+self.scale*h.mean(dim=1,keepdim=True)
        class Tiny(Fake):
            def __init__(self):
                self._model=SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([Block(.01),Block(.04),Block(.09),Block(.16)])))
                self.traces=[]
            def predict(self,ids,positions,probes):
                h=torch.tensor(ids,dtype=torch.float64)[None,:,None];trace=[]
                for layer in self._model.model.layers:h=layer(h);trace.append(h.clone())
                self.traces.append(trace)
                rows=[]
                for pos,ps in zip(positions,probes):
                    probs=torch.softmax(torch.linspace(-1,1,128,dtype=torch.float64)*h[0,pos,0]/100,dim=-1)
                    top=int(probs.argmax());rows.append(dict(token_id=top,confidence=float(probs[top]),probabilities={str(k):float(probs[k]) for k in ps}))
                return rows
        b=Tiny();builder=v22.CanvasBuilder(b);p=self.groups[0]['p'];target=builder.build(p,'downstream')
        jobs=[dict(mode='sustained',layers=[1,2,3],site='answer_A'),dict(mode='single',layers=[4],site='answer_A')]
        expected=v2.gold(p,'downstream');alternative={k:v+1 for k,v in expected.items()}
        src=builder.build(v22.source(p,'change_A',1),'downstream')
        context,rows=v22.evaluate(b,target,src,expected,alternative,jobs,'change_A')
        self.assertNotEqual(rows[0]['delta'],0);self.assertEqual(rows[1]['delta'],0)
        for l in range(3):self.assertTrue(torch.equal(b.traces[2][l][:,target['spans']['A']],b.traces[1][l][:,src['spans']['A']]))
        _,rows=v22.evaluate(b,target,target,expected,alternative,jobs,'self')
        self.assertTrue(all(r['delta']==0 for r in rows))
        self.assertTrue(all(not m._forward_hooks for m in b._model.model.layers))


if __name__=='__main__':unittest.main()
