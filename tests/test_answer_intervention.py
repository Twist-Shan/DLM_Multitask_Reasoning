import copy
import unittest
from dlm_coupling.backends.mock import MockTokenizer
from dlm_coupling.text_generation import generate_blockwise
from dlm_coupling.answer_intervention import answer_events,choose_fork,replay_state,continue_fork


class InterventionTests(unittest.TestCase):
    def fixture(self):
        class Tokenizer(MockTokenizer):
            eos_token_id=900
            def decode(self,ids,**kw):
                return ''.join('<eos>' if x==900 else super(Tokenizer,self).decode([x]) for x in ids)
        tok=Tokenizer();want=tok.encode('{"A":21,"B":29}')+[900]
        plen=len(tok.encode(tok.apply_chat_template([{'role':'user','content':'x'}])))
        source=want.index(22);target=want.index(30)
        class Backend:
            mask_id=0;tokenizer=tok
            def predict(self,tokens,positions,probes):
                result=[]
                for pos in positions:
                    j=pos-plen;value=want[j] if j<len(want) else 900
                    if j==target and tokens[plen+source]==23: value=31
                    result.append({'token_id':value,'token_text':'','confidence':0.9})
                return result
        backend=Backend();trace=generate_blockwise(backend,'x',64,32)
        return backend,{'prompt':'x','order':'AB','trace':trace,'answers':{'A':21,'B':29}}

    def test_replay_control_and_counterfactual(self):
        b,r=self.fixture();ev=answer_events(b.tokenizer,r)
        f,error=choose_fork(b.tokenizer,r,ev,'A','B');self.assertIsNone(error)
        prefix,base=replay_state(b,r,f);prefix2,changed=replay_state(b,r,f,True)
        self.assertEqual(prefix,prefix2)
        self.assertEqual([i for i in range(len(base)) if base[i]!=changed[i]],f['positions'])
        control=continue_fork(b,r,f);altered=continue_fork(b,r,f,True)
        self.assertEqual(control['output_token_ids'],r['trace']['output_token_ids'])
        self.assertEqual(altered['answers'],{'A':22,'B':30})

    def test_reject_target_committed_and_cross_block(self):
        b,r=self.fixture();ev=answer_events(b.tokenizer,r)
        self.assertEqual(choose_fork(b.tokenizer,r,ev,'B','A')[1],'target_already_started')
        ev=copy.deepcopy(ev);ev['answers']['B']['blocks']=[1]
        self.assertEqual(choose_fork(b.tokenizer,r,ev,'A','B')[1],'different_blocks')

    def test_ignore_numbers_in_reasoning(self):
        b,r=self.fixture();r['trace']['text']='<think>21'
        self.assertEqual(answer_events(b.tokenizer,r)['error'],'unfinished_reasoning')

if __name__=='__main__': unittest.main()
