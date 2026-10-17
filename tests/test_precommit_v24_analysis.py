"""Checks for analysis failure modes, without a GPU or model weights."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_precommit_v24 import analyze,estimate


class AnalysisTests(unittest.TestCase):
    def test_cluster_weighting_does_not_count_repeated_conditions_as_groups(self):
        result=estimate([('a',1),('a',3),('b',-1)])
        self.assertEqual(result['mean'],.5)
        self.assertEqual(result['groups'],2)
        self.assertEqual(result['observations'],3)
        self.assertEqual(result['ci95'],[-1,2])
        self.assertEqual(result,estimate([('a',1),('a',3),('b',-1)]))

    def test_incomplete_run_is_not_reported_as_complete(self):
        with patch.object(Path,'read_text',return_value=json.dumps(dict(status='failed'))):
            with self.assertRaisesRegex(ValueError,'Incomplete run'):analyze(Path('unused'))

    def test_complete_label_cannot_hide_missing_protocol_cells(self):
        names=('behavior','contexts','patch','rollout','space','space_contexts','space_patch')
        metadata=json.dumps(dict(status='complete',groups=30,outputs=dict.fromkeys(names,0)))
        with patch.object(Path,'read_text',side_effect=[metadata]+['']*len(names)):
            with self.assertRaisesRegex(ValueError,'Protocol count'):analyze(Path('unused'))


if __name__=='__main__':unittest.main()
