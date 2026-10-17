import unittest
from types import SimpleNamespace

from dlm_coupling import precommit_v24 as v
from test_precommit_v2 import Characters,Fake
from test_precommit import torch


class Tokenizer(Characters):
    eos_token_id=126


class Backend(Fake):
    tokenizer=Tokenizer()
    def __init__(self):self.calls=[]
    def predict(self,ids,positions,probes):
        self.calls.append((ids.copy(),positions.copy()))
        return [dict(token_id=ord('1'),confidence=.5,probabilities={str(p):.1 for p in ps})
                for ps in probes]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=v.dataset();cls.groups=cls.data['groups']

    def test_data_reuses_only_calibration_and_balances_value_equality(self):
        self.assertEqual(self.data,v.dataset())
        self.assertEqual(len(self.groups),30)
        self.assertEqual(sum(g['equal_AC'] for g in self.groups),15)
        self.assertTrue(all(g['base_case'].startswith('v2-calibration-') for g in self.groups))
        for g in self.groups:
            ex=v.gold(g['p'],'A')
            self.assertEqual(ex['A']==ex['C'],g['equal_AC'])

    def test_independent_sources_and_compensation_both_signs(self):
        for g in self.groups:
            p=g['p']
            for parent in ('A','C'):
                base=v.gold(p,parent)
                for variable in ('A','C'):
                    other='C' if variable=='A' else 'A'
                    for sign in (-1,1):
                        changed=v.gold(v.source(p,parent,'change_'+variable,sign),parent)
                        self.assertEqual(changed[variable],base[variable]+sign)
                        self.assertEqual(changed[other],base[other])
                        self.assertEqual(changed['B'],base['B']+(sign if variable==parent else 0))
                        self.assertEqual(v.gold(v.source(p,parent,'same_'+variable,sign),parent),base)
                        comp=v.gold(v.source(p,parent,'comp_'+variable,sign),parent)
                        self.assertEqual(comp['B'],base['B'])
                        self.assertEqual(comp[variable],base[variable]+sign)
                        bonly=v.gold(v.source(p,parent,'B_only',sign),parent)
                        self.assertEqual(bonly,dict(base,B=base['B']+sign))
        with self.assertRaises(ValueError):v.source(p,'A','change_A',0)
        with self.assertRaises(ValueError):v.source(p,'A','unknown',1)

    def test_canvas_geometry_domain_and_native_ending(self):
        b=Backend();builder=v.CanvasBuilder(b);p=self.groups[0]['p']
        a,c=(builder.build(p,'A',layout) for layout in v.LAYOUTS)
        self.assertEqual(a['spans']['B'],c['spans']['B'])
        self.assertEqual(len(a['ids']),len(c['ids']))
        for cc in (a,c):
            self.assertEqual([cc['ids'][p] for p in cc['suffix_positions']],cc['suffix_ids'])
            self.assertTrue(all(cc['ids'][p]==b.mask_id for ps in cc['spans'].values() for p in ps))
            for k in cc['maps']:
                self.assertEqual(len(cc['maps'][k]),90)
                self.assertEqual(cc['maps'][k][29],list(map(ord,'29')))
        for kind in v.specification()['kinds']:
            for sign in (-1,1):
                v.check_alignment(a,builder.build(v.source(p,'A',kind,sign),'A'))

    def test_space_matching_uses_actual_token_count_and_digit_offsets(self):
        b=Backend();builder=v.CanvasBuilder(b);p=self.groups[0]['p']
        cc=[builder.space(p,'A',name) for name in v.specification()['space_usage']]
        self.assertEqual(len({len(c['ids']) for c in cc}),1)
        self.assertEqual(len({tuple(c['spans']['B']) for c in cc}),1)
        self.assertEqual(len({tuple(c['patch_positions']) for c in cc}),1)
        self.assertEqual(len({tuple(c['suffix_positions']) for c in cc}),1)
        self.assertTrue(all(cc[2]['ids'][i]==b.mask_id for i in cc[2]['patch_positions']))
        self.assertTrue(all(cc[3]['ids'][i]==b.tokenizer.eos_token_id for i in cc[3]['patch_positions']))
        baseline=builder.space(p,'A','mask_0')
        for name in v.specification()['space_quantity']:
            c=builder.space(p,'A',name)
            self.assertEqual(c['spans']['B'],baseline['spans']['B'])
            self.assertEqual(len(c['ids'])-len(baseline['ids']),int(name.split('_')[1]))

    def test_B_first_holds_other_answers_and_only_B_leaves_them_masked(self):
        b=Backend();cc=v.CanvasBuilder(b).build(self.groups[0]['p'],'A','CBA')
        rr=v.decode(b,cc)
        self.assertEqual([w['position'] for w in rr['writes'][:2]],cc['spans']['B'])
        for w in rr['writes'][:2]:
            self.assertTrue(all(w['state'][p]==b.mask_id for k in ('A','C') for p in cc['spans'][k]))
        short=v.decode(b,cc,only_B=True)
        self.assertEqual(short['nfe'],2)
        self.assertTrue(all(short['ids'][p]==b.mask_id for k in ('A','C') for p in cc['spans'][k]))

    def test_decoder_does_not_constrain_digits(self):
        class Nonnumeric(Backend):
            def predict(self,ids,positions,probes):
                return [dict(token_id=ord('x'),confidence=.9,probabilities={}) for _ in positions]
        b=Nonnumeric();cc=v.CanvasBuilder(b).build(self.groups[0]['p'],'A','B',0)
        rr=v.decode(b,cc,only_B=True)
        self.assertIsNone(rr['values']['B']);self.assertFalse(rr['valid'])

    def test_fixed_grid_size(self):
        kinds=v.specification()['kinds']
        self.assertEqual(sum(len(v.schedule(k))*(1 if k=='self' else 2) for k in kinds),49)

    @unittest.skipIf(torch is None,'optional local torch; required before cloud inference')
    def test_per_layer_capture_refresh_last_layer_null_and_cleanup(self):
        class Block(torch.nn.Module):
            def forward(self,h):return h+.01*h.mean(dim=1,keepdim=True)
        class Tiny(Backend):
            def __init__(self):
                super().__init__()
                self._model=SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([Block() for _ in range(32)])))
            def predict(self,ids,positions,probes):
                self.calls.append((ids.copy(),positions.copy()))
                h=torch.tensor(ids,dtype=torch.float64)[None,:,None]
                for layer in self._model.model.layers:h=layer(h)
                out=[]
                for pos,ps in zip(positions,probes):
                    prob=torch.softmax(torch.arange(128,dtype=torch.float64)*h[0,pos,0]/10000,dim=-1)
                    out.append(dict(token_id=ord('1'),confidence=.5,probabilities={str(p):float(prob[p]) for p in ps}))
                return out
        b=Tiny();builder=v.CanvasBuilder(b);p=self.groups[0]['p']
        cc=builder.build(p,'A','CBA');ss=builder.build(v.source(p,'A','change_A',1),'A','CBA')
        jobs=[dict(site='A',mode=m,layers=v.MODES[m]) for m in ('L31','L32','L24_31')]
        _,rows=v.patch_scan(b,cc,ss,v.gold(p,'A'),1,jobs,'change_A')
        self.assertNotEqual(rows[0]['delta'],0);self.assertEqual(rows[1]['delta'],0)
        _,rows=v.patch_scan(b,cc,cc,v.gold(p,'A'),1,jobs,'self')
        self.assertTrue(all(r['delta']==0 for r in rows))
        rr=v.decode(b,cc,'B_first',ss,'persistent',cc['spans']['A'],v.MODES['L24_31'])
        self.assertEqual([w['patched'] for w in rr['writes']],[True,True,False,False,False,False])
        first,second=rr['writes'][:2]
        self.assertEqual(second['Source_state'][first['position']],first['token_id'])
        self.assertTrue(all(not layer._forward_hooks for layer in b._model.model.layers))
        with self.assertRaises(ValueError):
            v.patch_scan(b,cc,ss,v.gold(p,'A'),1,jobs,'change_A',regions=dict(A=cc['spans']['B']))


if __name__=='__main__':unittest.main()
