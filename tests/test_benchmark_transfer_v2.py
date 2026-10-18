import unittest
import importlib.util
from pathlib import Path
from dlm_coupling import benchmark_transfer_v2 as t
from dlm_coupling.text_generation import generate_blockwise


class Tokenizer:
    eos_token_id=99
    def apply_chat_template(self,*args,**kwargs): return 'p'
    def encode(self,*args,**kwargs): return [10]
    def decode(self,ids,**kwargs): return ','.join(map(str,ids))


class Backend:
    tokenizer=Tokenizer()
    mask_id=0
    def __init__(self): self.calls=[]
    def predict(self,ids,positions,probes):
        self.calls.append(ids.copy())
        return [dict(token_id=99 if p==4 else p,token_text=str(p),confidence=.9) for p in positions]


class V2Tests(unittest.TestCase):
    def test_analysis_pairs_only_same_sampler_and_same_case(self):
        spec=importlib.util.spec_from_file_location('analysis_v2',Path(__file__).resolve().parents[1]/'scripts/analyze_benchmark_transfer_v2.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def row(case,request,sampler,em):
            return dict(case_id=case,requested=request,sampler=sampler,benchmark='musique',format_valid=True,
                        scores={'B':{'em':em,'f1':em}},generation={'eos_found':True},
                        compute={'forwards':2,'processed_token_positions':100})
        rows=[row('a','B','block32',0),row('a','AB','block32',1),
              row('a','B','full',1),row('a','AB','full',0),row('b','AB','block32',1)]
        result=module.summarize(rows)
        self.assertEqual([(p['n'],p['delta_em']) for p in result['paired']],[(1,1),(1,-1)])
        with self.assertRaises(ValueError):module.summarize(rows+[rows[0]])

    def test_final_extraction(self):
        self.assertEqual(t.final_answer('<think>draft</think>\nFINAL:\nB: x')[0],'B: x')
        self.assertIsNone(t.final_answer('<think>FINAL:\nB: x')[0])
        self.assertIsNone(t.final_answer('FINAL:\nB: x\nFINAL:\nB: y')[0])
        self.assertEqual(t.final_answer('B: x')[0],'B: x')
        c=dict(benchmark='musique',answers={'B':['x']})
        self.assertEqual(t.score(c,'<think>B: x</think>FINAL:\nB: wrong','B')['scores']['B']['em'],0)

    def test_blockwise_batched_commits_stop_after_finished_eos_block(self):
        b=Backend();g=generate_blockwise(b,'x',max_new_tokens=12,block_size=4,tokens_per_step=3)
        self.assertEqual(g['nfe'],2)
        self.assertEqual(g['output_token_ids'],[1,2,3,99])
        self.assertTrue(g['eos_found'])
        self.assertTrue(all(len(ids)==5 for ids in b.calls))
        self.assertEqual([len(s['writes']) for s in g['steps']],[3,1])
        with self.assertRaises(ValueError):generate_blockwise(b,'x',tokens_per_step=0)


if __name__=='__main__':unittest.main()
