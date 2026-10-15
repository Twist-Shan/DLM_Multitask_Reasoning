import unittest

from dlm_coupling import precommit as v1, precommit_v2 as v2


class Characters:
    def encode(self,text,**kwargs):return list(map(ord,text))
    def decode(self,ids,**kwargs):return ''.join(map(chr,ids))
    def apply_chat_template(self,messages,**kwargs):
        prefix='<user>'+messages[0]['content']+'</user><assistant>'
        return prefix if len(messages)==1 else prefix+messages[1]['content']+'<native_end>\n'


class Fake:
    tokenizer=Characters();mask_id=5;family='illada'


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=v2.dataset()

    def test_additive_gold_source_compensation_and_counterfactual(self):
        for g in self.data['groups']:
            p=g['p']
            for rel in v2.RELATIONS:
                ex=v2.gold(p,rel)
                src=v2.source(p,rel,'diff_C_same_answers')
                self.assertEqual(v2.gold(src,rel),ex)
                self.assertEqual(v2.gold(v2.source(p,rel,'same_C_same_answers'),rel),ex)
                cf=v2.counterfactual(p,rel,'diff_C_same_answers')
                self.assertEqual(cf['A'],ex['A']+1)
                self.assertEqual(cf['B'],ex['B']+(0 if rel=='independent' else 1))
                self.assertNotIn('2 * ',v2.prompt(p,rel))
                self.assertEqual(ex['A'],p['x']+p['y']+p['u']+p['r'])
                self.assertEqual(ex['B'],(p['w']+p['z'] if rel=='independent' else
                    p['x']+p['y'] if rel=='shared' else ex['A'])+p['v']+p['s'])
        self.assertEqual(v2.source(p,'shared','diff_C_same_answers')['v'],p['v']-1)

    def test_new_stream_and_explicit_disjointness_including_sources(self):
        old=v1.dataset()
        seen=set().union(*(v2.variant_keys(g['p'],v1.source) for g in old['groups']))
        self.assertEqual(self.data,v2.dataset())
        self.assertEqual(len(self.data['groups']),160)
        self.assertNotEqual(self.data['streams']['calibration'],old['streams']['calibration'])
        for g in self.data['groups']:
            keys=v2.variant_keys(g['p'])
            self.assertFalse(keys&seen)
            seen.update(keys)

    def test_native_suffix_placement_and_legacy_replay(self):
        p=self.data['groups'][0]['p'];b=Fake()
        for task in ('legacy','additive'):
            builder=v2.CanvasBuilder(b,task)
            for req in ('AB','BA','A','B'):
                for unused in (False,True):
                    cc=builder.build(p,'shared',req,unused=unused)
                    self.assertEqual(cc['native_suffix'],'<native_end>\n')
                    self.assertTrue(all(cc['ids'][i]==b.mask_id for ps in cc['spans'].values() for i in ps))
                    self.assertEqual([cc['ids'][i] for i in cc['suffix_positions']],cc['suffix_ids'])
                    if unused:self.assertEqual(max(cc['unused'])+1,min(cc['suffix_positions']))
                    self.assertEqual(len(cc['maps']['A' if 'A' in req else 'B']),90)
                    if task=='legacy':
                        bare=builder.build(p,'shared',req,'numeric',unused,'bare')
                        orig=v1.CanvasBuilder(b).build(p,'shared',req,'numeric',unused)
                        for k in ('ids','spans','maps','unused','prompt'):self.assertEqual(bare[k],orig[k])

    def test_suffix_uses_template_not_hardcoded_model_token(self):
        class Other(Characters):
            def apply_chat_template(self,messages,**kwargs):
                return super().apply_chat_template(messages,**kwargs).replace('<native_end>','END_TWO')
        b=Fake();b.tokenizer=Other()
        cc=v2.CanvasBuilder(b).build(self.data['groups'][0]['p'],'shared')
        self.assertEqual(cc['native_suffix'],'END_TWO\n')

    def test_gate_ignores_joint_and_pools_canvas_but_rejects_duplicates(self):
        rows=[dict(case=f'c{i}',relation=rel,requested=req,unused=u,correct=True,valid=True)
              for i in range(30) for rel in v2.RELATIONS for req in ('A','B') for u in (False,True)]
        extra=[dict(case='joint',relation='shared',requested='AB',unused=True,correct=False,valid=False)]
        score=v2.score_gate(rows+extra)
        self.assertTrue(score['pass_gate'])
        self.assertEqual(len(score['cells']),6)
        self.assertEqual(score['cells']['shared/A']['outputs'],60)
        with self.assertRaises(ValueError):v2.score_gate(rows+[rows[0]])
        with self.assertRaises(ValueError):v2.score_gate(rows[:-1])
        self.assertGreater(v2.wilson_lower(18,20),.69)
        self.assertLess(v2.wilson_lower(17,20),.7)
        self.assertEqual(v2.cluster_interval(rows),[1.,1.])


if __name__=='__main__':unittest.main()
