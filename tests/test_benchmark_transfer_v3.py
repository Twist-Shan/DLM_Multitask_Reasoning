import unittest
from unittest.mock import patch
from dlm_coupling import benchmark_transfer_v3 as task


class ConcreteTemplateTests(unittest.TestCase):
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
