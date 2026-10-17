import unittest

from dlm_coupling import precommit_v2 as old,precommit_v25 as v
from test_precommit_v24 import Backend


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=v.dataset();cls.groups=cls.data['groups']

    def test_original_cases_and_answers_are_unchanged(self):
        self.assertEqual(self.groups,[g for g in old.dataset()['groups'] if g['split']=='calibration'])
        self.assertEqual(len(self.groups),30)
        self.assertEqual(v.dataset(),self.data)
        for g in self.groups:
            item=v.redundant(g['p'])
            for parent in ('A','R'):
                values=v.redundant_gold(item,parent)
                self.assertEqual(values['R'],values['A'])
                self.assertEqual({k:values[k] for k in ('A','B')},old.gold(g['p'],'downstream'))

    def test_original_token_canvas_is_identical(self):
        b=Backend();builder=v.CanvasBuilder(b);prior=old.CanvasBuilder(b)
        for relation in old.RELATIONS:
            p=self.groups[0]['p'];cc=builder.core(p,relation)
            previous=prior.build(p,relation,'AB','numeric',True,'native')
            for key in ('ids','spans','maps','unused','suffix_positions','native_suffix','prompt','output_scaffold'):
                self.assertEqual(cc[key],previous[key],key)

    def test_order_does_not_change_prompt_and_space_does_not_move_answers(self):
        b=Backend();builder=v.CanvasBuilder(b);p=self.groups[0]['p']
        ab,ba=(builder.core(p,'downstream',layout) for layout in ('AB','BA'))
        self.assertEqual(ab['prompt'],ba['prompt'])
        self.assertLess(ba['spans']['B'][0],ba['spans']['A'][0])
        for layout in ('AB','BA'):
            baseline=builder.core(p,'downstream',layout,'mask_0')
            for quantity in v.QUANTITIES:
                cc=builder.core(p,'downstream',layout,quantity)
                self.assertEqual(cc['spans'],baseline['spans'])
                self.assertEqual(cc['prompt'],baseline['prompt'])
                self.assertEqual(len(cc['ids'])-len(baseline['ids']),cc['extra'])
                self.assertEqual([cc['ids'][i] for i in cc['suffix_positions']],cc['suffix_ids'])
                expected=b.mask_id if cc['filler']=='mask' else b.tokenizer.eos_token_id
                self.assertTrue(all(cc['ids'][i]==expected for i in cc['unused']))

    def test_redundant_sources_have_independent_values_and_valid_compensation(self):
        for group in self.groups:
            item=v.redundant(group['p'])
            for parent in ('A','R'):
                target=v.redundant_gold(item,parent)
                for variable in ('A','R'):
                    other='R' if variable=='A' else 'A'
                    for sign in (-1,1):
                        changed=v.redundant_gold(v.redundant_source(item,parent,'change_'+variable,sign),parent)
                        self.assertEqual(changed[variable],target[variable]+sign)
                        self.assertEqual(changed[other],target[other])
                        self.assertEqual(changed['B'],target['B']+(sign if parent==variable else 0))
                        same=v.redundant_gold(v.redundant_source(item,parent,'same_'+variable,sign),parent)
                        self.assertEqual(same,target)
                        comp=v.redundant_gold(v.redundant_source(item,parent,'comp_'+variable,sign),parent)
                        self.assertEqual(comp['B'],target['B'])
                self.assertEqual(v.redundant_gold(v.redundant_source(item,parent,'B_only',1),parent),dict(target,B=target['B']+1))

    def test_redundancy_is_numeric_definition_not_an_alias_or_gold_value(self):
        item=v.redundant(self.groups[0]['p']);text=v.redundant_prompt(item,'A')
        original=old.prompt(item['p'],'downstream','AB','numeric')
        without_R='\n'.join(line for line in text.splitlines() if not line.startswith('Definition of R:'))
        self.assertEqual(without_R.replace('Requested outputs: A, B, R.','Requested outputs: A, B.'),original)
        self.assertNotIn('R = A',text)
        self.assertNotIn('R = '+str(v.redundant_gold(item,'A')['R'])+'.',text)
        self.assertEqual(v.redundant_prompt(item,'R').replace('B = R +','B = A +'),text)

    def test_alias_geometry_and_source_alignment(self):
        b=Backend();builder=v.CanvasBuilder(b);item=v.redundant(self.groups[0]['p'])
        for parent in ('A','R'):
            left,right=(builder.alias(item,parent,layout) for layout in ('ABR','RBA'))
            self.assertEqual(left['spans']['B'],right['spans']['B'])
            self.assertEqual(len(left['ids']),len(right['ids']))
            for layout in ('ABR','RBA'):
                cc=builder.alias(item,parent,layout)
                for kind in v.ALIAS_KINDS:
                    for sign in (-1,1):
                        v.check_alignment(cc,builder.alias(v.redundant_source(item,parent,kind,sign),parent,layout))

    def test_B_first_holds_original_A_masked_and_keeps_compute_fixed(self):
        b=Backend();builder=v.CanvasBuilder(b);p=self.groups[0]['p']
        for quantity in v.QUANTITIES:
            cc=builder.core(p,'downstream','BA',quantity)
            result=v.decode(b,cc,'B_first')
            self.assertEqual(result['nfe'],4)
            for step in result['writes'][:2]:
                self.assertIn(step['position'],cc['spans']['B'])
                self.assertTrue(all(step['state'][i]==b.mask_id for i in cc['spans']['A']))
            self.assertTrue(all(result['ids'][i]==cc['ids'][i] for i in cc['unused']))

    def test_fixed_counts(self):
        spec=v.specification();planned=spec['planned_per_group']
        self.assertEqual(len(v.core_conditions())*2,planned['core_behavior'])
        contexts=sum(sum(1 if k=='self' else 2 for k in v.core_kinds(rel)) for rel in ('downstream','independent'))*2
        patches=sum(sum(len(v.core_schedule(k))*(1 if k=='self' else 2) for k in v.core_kinds(rel)) for rel in ('downstream','independent'))*2
        self.assertEqual(contexts,planned['core_contexts']);self.assertEqual(patches,planned['core_patch'])
        self.assertEqual(sum(len(v.alias_schedule(k))*(1 if k=='self' else 2) for k in v.ALIAS_KINDS)*4,planned['alias_patch'])


if __name__=='__main__':unittest.main()
