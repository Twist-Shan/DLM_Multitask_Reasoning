import unittest

from dlm_coupling import precommit as v1, precommit_v2 as v2, precommit_v21 as v21
from test_precommit_v2 import Fake, Characters
from test_precommit import torch


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.groups=[g for g in v2.dataset()['groups'] if g['split']=='calibration']

    def test_factorial_decouples_text_and_slots_and_preserves_v2(self):
        b=Fake();builder=v21.CanvasBuilder(b);old=v2.CanvasBuilder(b);p=self.groups[1]['p']
        for req in ('B','AB'):
            for slots in ('B','AB'):
                for unused in (False,True):
                    cc=builder.build(p,req,slots,unused)
                    self.assertEqual(cc['prompt'],v2.prompt(p,'downstream',req))
                    self.assertEqual(set(cc['spans']),set(slots))
                    self.assertEqual(b.tokenizer.decode([cc['ids'][i] for i in cc['prompt_A']]),cc['prompt_A_text'])
                    self.assertTrue(all(cc['ids'][i]==b.mask_id for ps in cc['spans'].values() for i in ps))
                    self.assertEqual([cc['ids'][i] for i in cc['suffix_positions']],cc['suffix_ids'])
                    if req==slots:
                        original=old.build(p,'downstream',req,unused=unused)
                        for k in ('ids','spans','maps','unused','prompt','suffix_ids'):self.assertEqual(cc[k],original[k])
        self.assertEqual(builder.build(p,'B','AB')['spans'],builder.build(p,'B','AB')['spans'])
        self.assertNotEqual(builder.build(p,'B','AB')['prompt'],builder.build(p,'AB','AB')['prompt'])

    def test_both_signs_and_same_A_controls_all_groups(self):
        builder=v21.CanvasBuilder(Fake())
        for g in self.groups:
            p=g['p'];expected=v2.gold(p,'downstream');target=builder.build(p)
            for sign in (-1,1):
                for kind in v21.SOURCE_KINDS:
                    q=v21.source(p,kind,sign);actual=v2.gold(q,'downstream')
                    self.assertEqual(actual,{k:v+(sign if kind=='change_A' else 0) for k,v in expected.items()})
                    self.assertEqual(q['v']+q['s'],p['v']+p['s'])
                    v21.check_alignment(target,builder.build(q))
        with self.assertRaises(ValueError):v21.source(p,'change_A',0)

    def test_exact_context_gate_ignores_single_condition_and_bad_other_cells(self):
        rows=[dict(case=g['id'],requested='AB',slots='AB',unused=True,
            t0=dict(values=dict(A=20,B=30),valid=True),expected=dict(A=20,B=30)) for g in self.groups]
        extra=dict(case='single',requested='B',slots='B',unused=True,t0={},expected={})
        self.assertTrue(v21.baseline_gate(rows+[extra])['pass_gate'])
        for r in rows[:14]:r['t0']['values']['B']=31
        self.assertTrue(v21.baseline_gate(rows)['pass_gate'])
        for r in rows[14:16]:r['t0']['values']['B']=31
        self.assertFalse(v21.baseline_gate(rows)['pass_gate'])
        with self.assertRaises(ValueError):v21.baseline_gate(rows+[rows[0]])

    def test_region_alignment_rejects_shifted_prompt(self):
        builder=v21.CanvasBuilder(Fake());cc=builder.build(self.groups[0]['p'])
        with self.assertRaises(ValueError):v21.check_alignment(cc,{**cc,'prompt_A':[i+1 for i in cc['prompt_A']]})

    def test_prompt_period_newline_bpe_merge_preserves_answer_boundaries(self):
        class Merged(Characters):
            def encode(self,text,**kwargs):return super().encode(text.replace('.\n','\u20ac'))
            def decode(self,ids,**kwargs):return super().decode(ids).replace('\u20ac','.\n')
        b=Fake();b.tokenizer=Merged();p=self.groups[1]['p'];cc=v21.CanvasBuilder(b).build(p)
        self.assertTrue(cc['prompt_A_text'].endswith('.\n'))
        self.assertEqual(b.tokenizer.decode([cc['ids'][i] for i in cc['prompt_A']]),cc['prompt_A_text'])
        old=v2.CanvasBuilder(b).build(p,'downstream',unused=True)
        for key in ('ids','maps','spans','unused'):self.assertEqual(cc[key],old[key])

    @unittest.skipIf(torch is None,'optional CPU torch hook test')
    def test_four_region_hooks_causal_effect_and_nulls(self):
        from types import SimpleNamespace
        class Block(torch.nn.Module):
            def forward(self,h):return h+.1*h.mean(dim=1,keepdim=True)
        class Tiny(Fake):
            def __init__(self):
                self._model=SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([Block() for _ in range(4)])))
            def predict(self,ids,positions,probes):
                h=torch.tensor(ids,dtype=torch.float64)[None,:,None]
                for layer in self._model.model.layers:h=layer(h)
                rows=[]
                for pos,ps in zip(positions,probes):
                    probs=torch.softmax(torch.linspace(-1,1,128,dtype=torch.float64)*h[0,pos,0]/100,dim=-1)
                    top=int(probs.argmax())
                    rows.append(dict(token_id=top,confidence=float(probs[top]),
                        probabilities={str(k):float(probs[k]) for k in ps}))
                return rows
        b=Tiny();builder=v21.CanvasBuilder(b);p=self.groups[1]['p'];target=builder.build(p)
        for kind in v21.SOURCE_KINDS:
            src=builder.build(v21.source(p,kind,1))
            rows=list(v21.scan(b,target,src,v2.gold(p,'downstream'),1,[0,3],kind))
            self.assertEqual(len(rows),8)
            self.assertEqual({r['site'] for r in rows},set(v21.SITES))
            self.assertTrue(all(r['delta']==0 for r in rows if r['layer']==4))
            if kind=='change_A':self.assertTrue(any(abs(r['delta'])>0 for r in rows if r['layer']==1))
            if kind=='self':self.assertTrue(all(r['patch_l2']==0 for r in rows if r['site']!='A_to_unused'))
            self.assertTrue(all(not m._forward_hooks for m in b._model.model.layers))


if __name__=='__main__':unittest.main()
