import contextlib
import unittest

from dlm_coupling.precommit import (CanvasBuilder, RELATIONS, SOURCE_KINDS, dataset,
    source, gold, counterfactual, prompt, check_alignment, decode, log_ratio,
    probes_for, score_gate, layer_grid)
from dlm_coupling.backends.base import raw_logit_positions
try:
    import torch
except ImportError:
    torch=None


class Characters:
    def encode(self,text,**kwargs): return list(map(ord,text))
    def decode(self,ids,**kwargs): return ''.join(map(chr,ids))
    def apply_chat_template(self,messages,**kwargs): return '<user>'+messages[0]['content']+'</user><assistant>'


class Numbers(Characters):
    """Whole two-digit numbers are one token; surrounding text stays unchanged."""
    def encode(self,text,**kwargs):
        import re
        return [1000+int(t) if len(t)==2 and t.isdigit() else ord(t)
                for t in re.findall(r'\d{2}|.',text,re.S)]
    def decode(self,ids,**kwargs):
        return ''.join(str(i-1000) if i>=1000 else chr(i) for i in ids)


class Fake:
    tokenizer=Characters();mask_id=5;family='illada'
    changed=False
    def predict(self,ids,positions,probes):
        return [dict(token_id=ord('3') if self.changed else ord('2'),
                     confidence=(1/(p+1) if self.changed else p/10000)) for p in positions]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ds=dataset()

    def test_reproducible_disjoint_groups_and_sources(self):
        self.assertEqual(self.ds,dataset())
        self.assertEqual(len(self.ds['groups']),140)
        seen=set()
        for group in self.ds['groups']:
            p=group['p'];keys=set()
            for rel in RELATIONS:
                self.assertEqual(gold(p,rel),gold(source(p,rel,'diff_C_same_answers'),rel))
                self.assertEqual(gold(p,rel),gold(source(p,rel,'same_C_same_answers'),rel))
                cf=counterfactual(p,rel,'diff_C_same_answers')
                self.assertEqual(cf['A'],gold(p,rel)['A']+1)
                self.assertEqual(cf['B'],gold(p,rel)['B']+(0 if rel=='independent' else 2))
                for kind in SOURCE_KINDS:keys.add(tuple(sorted(source(p,rel,kind).items())))
            self.assertFalse(keys&seen);seen.update(keys)

    def test_full_domain_audit_for_single_and_multi_token_numbers(self):
        p=self.ds['groups'][0]['p']
        for tokenizer,width in [(Characters(),2),(Numbers(),1)]:
            b=Fake();b.tokenizer=tokenizer;builder=CanvasBuilder(b)
            cc=builder.build(p,'shared',unused=True)
            self.assertEqual(len(cc['spans']['A']),width)
            self.assertEqual(len(cc['unused']),width)
            self.assertEqual(len(cc['maps']['B']),90)
            self.assertFalse(cc['uses_gold_to_prepare'])
            for ps in cc['spans'].values():self.assertTrue(all(cc['ids'][i]==5 for i in ps))
            for kind in SOURCE_KINDS:
                check_alignment(cc,builder.build(source(p,'shared',kind),'shared',unused=True))
            self.assertNotIn('C =',cc['prompt'])

    def test_format_boundary_failure_is_not_silently_filtered(self):
        class Broken(Numbers):
            def encode(self,text,**kwargs):
                return Characters().encode(text) if '99' in text else super().encode(text)
        b=Fake();b.tokenizer=Broken()
        with self.assertRaisesRegex(ValueError,'domain-wide'):
            CanvasBuilder(b).build(self.ds['groups'][0]['p'],'shared')

    def test_fixed_schedule_forces_positions_not_baseline_tokens(self):
        b=Fake();cc=CanvasBuilder(b).build(self.ds['groups'][0]['p'],'shared',unused=True)
        original=cc['ids'].copy();baseline=decode(b,cc)
        order=[w['position'] for w in baseline['writes']]
        self.assertEqual(decode(b,cc,fixed_order=order)['ids'],baseline['ids'])
        @contextlib.contextmanager
        def patch():
            b.changed=True
            try:yield
            finally:b.changed=False
        rr=decode(b,cc,fixed_order=order,patch_context=patch)
        self.assertEqual([w['position'] for w in rr['writes']],order)
        self.assertEqual(rr['writes'][0]['token_id'],ord('3'))
        self.assertEqual(sum(w['patched'] for w in rr['writes']),1)
        self.assertTrue(all(rr['ids'][i]==5 for i in cc['unused']))
        self.assertEqual(cc['ids'],original)
        with self.assertRaises(ValueError):decode(b,cc,fixed_order=order[:-1])

    def test_joint_and_separate_share_all_facts_and_definitions(self):
        p=self.ds['groups'][0]['p']
        texts=[prompt(p,'downstream',r).split('Requested outputs:')[0] for r in ('AB','BA','A','B')]
        self.assertEqual(len(set(texts)),1)
        self.assertIn('2 * A',texts[0])
        self.assertEqual(gold(dict(q=23),'copy'),dict(A=23,B=23))

    def test_gate_checks_each_relation_order_and_unused_condition(self):
        rows=[dict(style='symbolic',relation=rel,requested=req,unused=unused,correct=True,valid=True)
              for rel in RELATIONS for req in ('AB','BA','A','B') for unused in (False,True)]
        self.assertEqual(score_gate(rows,['symbolic'])['selected_style'],'symbolic')
        rows[-1]['correct']=False
        self.assertIsNone(score_gate(rows,['symbolic'])['selected_style'])

    def test_query_metric_and_dream_readout_are_explicit(self):
        p=self.ds['groups'][0]['p'];b=Fake();cc=CanvasBuilder(b).build(p,'shared')
        expected=gold(p,'shared');alt=dict(A=expected['A']+1,B=expected['B']+2)
        ps,probes=probes_for(cc,expected,alt)
        preds=[dict(probabilities={str(t):.2 for t in ts}) for ts in probes]
        self.assertEqual(log_ratio(cc,preds,ps,'A',expected['A'],alt['A']),0.)
        self.assertEqual(raw_logit_positions(cc['spans']['A'],'dream'),[p-1 for p in cc['spans']['A']])
        self.assertEqual(layer_grid(32),[7,15,23,30,31])
        self.assertEqual(layer_grid(28),[6,13,20,26,27])

    @unittest.skipIf(torch is None,'optional CPU torch hook test')
    def test_masked_region_scan_real_hooks_and_spatial_null(self):
        from types import SimpleNamespace
        from dlm_coupling.precommit import scan
        class Block(torch.nn.Module):
            def forward(self,h):return h+.1*h.mean(dim=1,keepdim=True)
        class Tiny(Fake):
            def __init__(self,family):
                self.family=family
                self._model=SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([Block() for _ in range(4)])))
            def predict(self,ids,positions,probes):
                h=torch.tensor(ids,dtype=torch.float64)[None,:,None]
                for layer in self._model.model.layers:h=layer(h)
                rows=[]
                for pos,ps in zip(raw_logit_positions(positions,self.family),probes):
                    probs=torch.softmax(torch.linspace(-1,1,128,dtype=torch.float64)*h[0,pos,0]/100,dim=-1)
                    top=int(probs.argmax())
                    rows.append(dict(token_id=top,confidence=float(probs[top]),
                                     probabilities={str(k):float(probs[k]) for k in ps}))
                return rows
        p=self.ds['groups'][0]['p']
        for family in ('illada','dream'):
            b=Tiny(family);cb=CanvasBuilder(b)
            target=cb.build(p,'shared',unused=True)
            src=cb.build(source(p,'shared','C_only'),'shared',unused=True)
            ex=gold(p,'shared');alt=dict(A=ex['A']+1,B=ex['B']+2)
            records=scan(b,target,src,'BA',ex,alt,[0,3])
            self.assertTrue(any(abs(r['delta'])>0 for r in records if r['layer']==1))
            self.assertTrue(all(r['delta']==0 for r in records if r['layer']==4))
            rr=scan(b,target,target,'BA',ex,alt,[0,3])
            self.assertTrue(all(r['predictions']==r['baseline'] for r in rr if r['site']=='answer_B'))
            self.assertTrue(all(not m._forward_hooks for m in b._model.model.layers))


if __name__=='__main__':unittest.main()
