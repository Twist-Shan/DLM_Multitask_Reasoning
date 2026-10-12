import unittest
from types import SimpleNamespace

from dlm_coupling.backends.mock import MockTokenizer
from dlm_coupling.tasks.igsm import evaluate_graph, relation, parse_answers, render
from dlm_coupling.text_generation import generate_text


class IGSMTests(unittest.TestCase):
    def graph(self):
        return {'x': {'expression': {'literal': 20}},
                'u': {'expression': {'op':'sum','args':[{'ref':'x'},{'literal':5}]}},
                'a': {'expression': {'op':'mul','args':[{'ref':'u'},{'literal':3}]}},
                'b': {'expression': {'op':'diff','args':[{'ref':'u'},{'literal':5}]}},
                'z': {'expression': {'literal':8}}}

    def test_modular_evaluation_and_relations(self):
        nodes=self.graph()
        self.assertEqual(evaluate_graph(nodes),{'x':20,'u':2,'a':6,'b':20,'z':8})
        self.assertEqual(relation(nodes,'a','b'),'shared')
        self.assertEqual(relation(nodes,'u','a'),'downstream')
        self.assertEqual(relation(nodes,'a','u'),'reverse_downstream')
        self.assertEqual(relation(nodes,'a','z'),'independent')
        nodes['b']['expression']={'op':'sum','args':[{'ref':'x'},{'literal':1}]}
        nodes['a']['expression']={'op':'sum','args':[{'ref':'x'},{'literal':2}]}
        self.assertEqual(relation(nodes,'a','b'),'shared_constants_only')

    def test_cycles_rejected(self):
        nodes=self.graph();nodes['x']['expression']={'ref':'a'}
        with self.assertRaisesRegex(ValueError,'Cyclic'): evaluate_graph(nodes)

    def test_strict_parser(self):
        self.assertEqual(parse_answers('{"A":12,"B":0}','AB'),({'A':12,'B':0},None))
        self.assertEqual(parse_answers('```json\n{"A":22}\n```','A'),({'A':22},None))
        for text in ['{"A":true}','{"A":23}','{"A":1,"A":2}','{"B":1}',
                     'answer: {"A":1}','{"A":1.0}','{"A":1} trailing','{"A":"12"}']:
            self.assertIsNotNone(parse_answers(text,'A')[1],text)

    def test_prompt_uses_no_gold(self):
        pair={'facts':'x=3','targets':{'A':{'question':'What is x?','gold':3}}}
        original=render(pair,'A');pair['targets']['A']['gold']=17
        self.assertEqual(render(pair,'A'),original)
        self.assertIn('modulo 23',original)

    def test_full_sequence_multi_token_and_eos(self):
        class Tokenizer(MockTokenizer):
            eos_token_id=900
            def decode(self,ids,**kwargs):
                return ''.join('<eos>' if t==900 else super(Tokenizer,self).decode([t]) for t in ids)
        tok=Tokenizer()
        desired=tok.encode('{"A":12}')+[900,1000+ord('!')]
        class Backend:
            mask_id=0
            tokenizer=tok
            model_config=SimpleNamespace(max_position_embeddings=1000)
            def predict(self,tokens,positions,probes):
                start=len(tokens)-len(desired)
                return [{'token_id':desired[i-start],'token_text':tok.decode([desired[i-start]]),
                         'confidence':0.8} for i in positions]
        result=generate_text(Backend(),'question',len(desired),tokens_per_step=2)
        self.assertEqual(result['text'],'{"A":12}')
        self.assertTrue(result['eos_found'])
        self.assertIn('<eos>!',result['raw_text'])
        self.assertEqual(result['nfe'],(len(desired)+1)//2)
        writes=[w['output_position'] for s in result['steps'] for w in s['writes']]
        self.assertEqual(writes,list(range(len(desired))))


if __name__=='__main__': unittest.main()
