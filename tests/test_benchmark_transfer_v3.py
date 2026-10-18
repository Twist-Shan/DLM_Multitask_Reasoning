import unittest
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
from dlm_coupling import benchmark_transfer_v3 as task


class ConcreteTemplateTests(unittest.TestCase):
    def test_expansion_is_balanced_and_preserves_diagnostic(self):
        scripts=Path(__file__).resolve().parents[1]/'scripts'
        sys.path.insert(0,str(scripts))
        spec=importlib.util.spec_from_file_location('official_diagnostic_test',scripts/'run_benchmark_official_diagnostic.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        data={'cases':[dict(id=f'{b}-{h}-{i}',benchmark=b,hops=h)
                       for b in ('musique','stepgame') for h in (2,3,4) for i in range(3)]}
        with patch.object(task,'prompt',side_effect=lambda c,r:c['id']+r):
            cases,jobs=runner.make_jobs(data,task,'expanded')
            self.assertEqual((len(cases),len(jobs)),(18,72))
            self.assertEqual(len({(j['case_id'],j['requested']) for j in jobs}),72)
            for c in cases:
                self.assertEqual([j['requested'] for j in jobs if j['case_id']==c['id']],['B','AB','BA','RB'])
            small,checks=runner.make_jobs(data,task,'diagnostic')
            self.assertEqual((len(small),len(checks)),(2,4))
            self.assertTrue(all(c['hops']==2 for c in small))

    def test_labels_and_order_without_changing_context(self):
        original = 'Private context remains exact. as LABEL: short answer.'
        for requested in ('B', 'AB', 'BA', 'RB'):
            with self.subTest(requested=requested), patch.object(task.previous, 'prompt', return_value=original):
                text = task.prompt({}, requested)
                self.assertTrue(text.startswith('Private context remains exact.'))
                self.assertNotIn('LABEL', text)
                self.assertEqual(text.split('FINAL:\n')[1].splitlines(),
                                 [f'{x}: [answer to question {x}]' for x in requested])


if __name__ == '__main__':
    unittest.main()
