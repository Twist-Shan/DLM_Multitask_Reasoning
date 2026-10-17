import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_precommit_v25 import analyze,correct
from dlm_coupling.precommit_v25 import specification


class Tests(unittest.TestCase):
    def test_t0_and_final_are_scored_separately(self):
        row=dict(expected=dict(A=22,B=31),t0=dict(values=dict(A=22,B=31)),values=dict(A=22,B=30))
        self.assertTrue(correct(row,'t0'));self.assertFalse(correct(row,'final'))
        row['values']['B']=None
        self.assertFalse(correct(row,'final'))

    def test_failed_run_cannot_be_summarized(self):
        with patch.object(Path,'read_text',return_value=json.dumps(dict(status='failed'))):
            with self.assertRaisesRegex(ValueError,'Incomplete run'):analyze(Path('unused'))

    def test_complete_flag_does_not_hide_missing_cells(self):
        names=specification()['planned_per_group']
        manifest=json.dumps(dict(status='complete',groups=30,outputs=dict.fromkeys(names,0)))
        with patch.object(Path,'read_text',side_effect=[manifest]+['']*len(names)):
            with self.assertRaisesRegex(ValueError,'Row count'):analyze(Path('unused'))


if __name__=='__main__':unittest.main()
