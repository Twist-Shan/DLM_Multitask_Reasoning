import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_precommit_v26 import analyze,audit_behavior,estimate
from dlm_coupling.precommit_v26 import attention_plan


class Tests(unittest.TestCase):
    def test_group_weighting_and_constant_effect_interval(self):
        a=estimate([('a',0)]*10+[('b',2)])
        self.assertEqual(a['mean'],1);self.assertEqual(a['groups'],2)
        self.assertEqual(estimate([('a',3),('b',3)])['ci95'],[3,3])

    def test_incomplete_run_is_not_summarized(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'manifest.json').write_text(json.dumps(dict(status='failed')))
            with self.assertRaisesRegex(ValueError,'Incomplete'):analyze(p)

    def test_attention_audit_rejects_wrong_direction_and_premature_parent(self):
        cc=dict(ids=[99]*6+[7],mask_id=99,spans={'A':[0,1],'B':[2,3]},
                unused=[4,5],suffix_positions=[6],suffix_ids=[7])
        state=cc['ids'].copy();writes=[]
        for p in (2,3,0,1):
            state[p]=1;writes.append(dict(position=p,state=state.copy()))
        row=dict(canvas=cc,nfe=4,forwards=4,writes=writes,policy='B_first',
                 attention_mode='tail_to_original',hook_forward_count=9,
                 attention=dict(attention_plan(cc,'tail_to_original'),calls_per_layer=[9]*32))
        audit_behavior(row)
        broken=copy.deepcopy(row);broken['attention']['query_positions']=[4,5]
        with self.assertRaisesRegex(ValueError,'Wrong attention'):audit_behavior(broken)
        broken=copy.deepcopy(row);broken['writes'][0]['state'][0]=1
        with self.assertRaisesRegex(ValueError,'Visible parent'):audit_behavior(broken)


if __name__=='__main__':unittest.main()
