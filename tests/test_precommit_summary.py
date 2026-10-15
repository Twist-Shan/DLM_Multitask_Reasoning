"""Check paired statistical units and prevent pre-existing hits counting as recovery."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('precommit_summary',Path(__file__).resolve().parents[1]/'scripts/summarize_precommit.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Tests(unittest.TestCase):
    def rows(self):
        rows=[]
        for case in range(2):
            for rel,extra in [('independent',1),('shared',4),('downstream',7),('copy',9)]:
                for site,delta in [('answer_B',5+case*2+extra),('unused',5)]:
                    rows.append(dict(kind='copy_scan' if rel=='copy' else 'scan',case=str(case),
                        relation=rel,requested='AB',direction='BA',source_kind='diff_C_same_answers',
                        layer=16,site=site,delta=delta,eligible=case==0))
        return rows

    def test_region_and_relation_contrasts_are_paired_by_underlying_group(self):
        result=module.summarize(self.rows())
        shared=next(r for r in result['cells'] if r['relation']=='shared')
        self.assertEqual((shared['groups'],shared['mean_excess_delta']), (2,5))
        self.assertEqual((shared['eligible_groups'],shared['eligible_mean_excess_delta']),(1,4))
        contrasts={r['contrast']:r for r in result['relation_contrasts']}
        self.assertEqual(contrasts['shared minus independent']['mean'],3)
        self.assertEqual(contrasts['shared minus independent']['descriptive_paired_bootstrap95'],[3,3])
        self.assertEqual(contrasts['downstream minus independent']['mean'],6)
        self.assertEqual(len(result['copy_cells']),1)
        self.assertEqual(result['copy_cells'][0]['groups'],2)
        self.assertEqual(result['copy_control_records'],4)

    def test_missing_and_duplicate_sites_fail_explicitly(self):
        rows=self.rows()
        with self.assertRaisesRegex(ValueError,'Incomplete'):module.summarize(rows[:-1])
        with self.assertRaisesRegex(ValueError,'Duplicate'):module.summarize(rows+[rows[0]])

    def test_baseline_counterfactual_is_not_counted_as_new_recovery(self):
        common=dict(kind='rollout',relation='shared',requested='AB',direction='BA',
            source_kind='diff_C_same_answers',site='answer_B',schedule='native',
            valid=True,expected_cf=dict(A=24,B=35),values=dict(A=24,B=33))
        result=module.summarize([dict(common,baseline_values=dict(A=24,B=33)),
                                 dict(common,baseline_values=dict(A=23,B=33))])
        row=result['rollouts'][0]
        self.assertEqual((row['n'],row['cf'],row['new_cf'],row['changed']),(2,2,1,1))
        self.assertEqual((row['other_changed'],row['joint_cf']),(0,0))


if __name__=='__main__':unittest.main()
