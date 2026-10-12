import unittest
from types import SimpleNamespace
from dlm_coupling.shared_causal import examples, answers, canvas, substitute, read_values, fill


class CharacterTokenizer:
    def encode(self, text, **kwargs): return list(map(ord, text))
    def decode(self, ids, **kwargs): return ''.join(map(chr, ids))
    def apply_chat_template(self, messages, **kwargs): return messages[0]['content']+'\nAssistant: '


class SharedCausalTests(unittest.TestCase):
    def test_counterfactual_not_donor_and_same_c_control(self):
        for row in examples(30):
            self.assertEqual(row['gold']['C'], answers(row['same'])['C'])
            self.assertEqual(row['hybrid']['C'], answers(row['donor'])['C'])
            for key in ('A','B'):
                self.assertNotEqual(row['hybrid'][key], answers(row['donor'])[key])
                self.assertNotEqual(row['hybrid'][key], row['gold'][key])
        self.assertEqual(examples(), examples(seed=123))

    def test_canvas_and_replacement_only_touch_c(self):
        b=SimpleNamespace(tokenizer=CharacterTokenizer(),mask_id=1)
        p=examples()[0]['base']
        ids,ss,_,_=canvas(b,p)
        original=ids.copy()
        substitute(b,ids,ss['C'],12)
        self.assertEqual([i for i,(a,c) in enumerate(zip(ids,original)) if a!=c],ss['C'])
        self.assertTrue(all(ids[i]==1 for i in ss['A']+ss['B']))
        self.assertEqual(read_values(b,ids,ss)['values']['C'],12)
        with self.assertRaises(ValueError): substitute(b,ids,ss['C'],123)

    def test_full_vocabulary_no_numeric_filter_and_input_immutable(self):
        b=SimpleNamespace(mask_id=1)
        b.predict=lambda ids,ps,probes:[dict(token_id=ord('X'),confidence=.9,token_text='X') for p in ps]
        original=[1,2,1]
        result,writes=fill(b,original,[0,2])
        self.assertEqual(original,[1,2,1])
        self.assertEqual(result,[88,2,88])
        self.assertEqual([w['position'] for w in writes],[0,2])


if __name__=='__main__': unittest.main()
