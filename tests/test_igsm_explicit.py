import copy
import unittest
from dlm_coupling.tasks.igsm import evaluate_graph
from dlm_coupling.tasks.igsm_explicit import facts_text,parse_answers,render
from dlm_coupling.text_generation import generate_blockwise
from dlm_coupling.backends.mock import MockTokenizer


class ExplicitTests(unittest.TestCase):
    def test_blockwise_stops_only_after_completing_eos_block(self):
        class Tokenizer(MockTokenizer):
            eos_token_id=900
            def decode(self,ids,**kwargs):
                return ''.join('<eos>' if t==900 else super(Tokenizer,self).decode([t]) for t in ids)
        tok=Tokenizer();want=tok.encode('{"A":28}')+[900]
        plen=len(tok.encode(tok.apply_chat_template([{'role':'user','content':'x'}],tokenize=False,add_generation_prompt=True)))
        class Backend:
            mask_id=0
            tokenizer=tok
            def predict(self,tokens,positions,probes):
                return [{'token_id':want[p-plen] if p-plen<len(want) else 900,
                         'token_text':'','confidence':0.8} for p in positions]
        r=generate_blockwise(Backend(),'x',32,4)
        self.assertEqual(r['text'],'{"A":28}')
        self.assertEqual(r['nfe'],((len(want)+3)//4)*4)
        self.assertEqual(r['output_budget'],32)
        self.assertTrue(r['eos_found'])
    def test_integer_arithmetic_differs_from_modular(self):
        nodes={'x':{'expression':{'literal':20}},'y':{'expression':{'op':'sum','args':[{'ref':'x'},{'literal':8}]}}}
        self.assertEqual(evaluate_graph(nodes)['y'],5)
        self.assertEqual(evaluate_graph(nodes,modulus=None)['y'],28)
        self.assertEqual(facts_text(nodes),'Q1 = 20\nQ2 = (Q1 + 8)')

    def test_final_answer_boundaries(self):
        self.assertEqual(parse_answers('<think>28</think>\nFINAL_JSON: {"A":28}','A'),({'A':28},None))
        self.assertEqual(parse_answers('Computed 28.\nFINAL_JSON: {"A":28}','A'),({'A':28},None))
        self.assertEqual(parse_answers('{"A":28}','A'),({'A':28},None))
        for t in ['<think>FINAL_JSON: {"A":28}', 'FINAL_JSON: {"A":28,"A":29}',
                  'FINAL_JSON: {"A":true}','FINAL_JSON: {"A":28} extra',
                  'Reasoning {"A":28}', 'FINAL_JSON: {"A":28,"B":1}']:
            self.assertIsNotNone(parse_answers(t,'A')[1],t)

    def test_prompt_gold_isolation_and_single_joint_facts(self):
        p={'nodes':{'x':{'expression':{'literal':3}},'y':{'expression':{'literal':5}}},
           'facts':'Q1 = 3\nQ2 = 5','targets':{'A':{'node':'x','gold':3},'B':{'node':'y','gold':5}}}
        original=render(p,'AB');p['targets']['A']['gold']=999
        self.assertEqual(original,render(p,'AB'))
        self.assertIn(p['facts'],render(p,'A'))
        self.assertIn('B: What is Q2?\nA: What is Q1?',render(p,'BA'))

if __name__=='__main__': unittest.main()
