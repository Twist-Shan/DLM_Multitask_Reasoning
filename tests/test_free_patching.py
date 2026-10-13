"""Audit reconstruction of out-of-order commitment without future-token leakage."""
import importlib.util
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('free_patch',Path(__file__).resolve().parents[1]/'scripts/run_free_reasoning_patching.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Tokenizer:
    def apply_chat_template(self,*args,**kwargs): return 'P'
    def encode(self,text,**kwargs): return [ord(c) for c in text]
    def decode(self,ids,**kwargs): return ''.join(chr(i) for i in ids)

class Backend:
    tokenizer=Tokenizer()
    mask_id=-1

class Tests(unittest.TestCase):
    def test_replay_preserves_masks_and_excludes_future_blocks(self):
        row=dict(prompt='ignored',steps=[
            dict(writes=[dict(output_position=2,token_id=65)]),
            dict(writes=[dict(output_position=0,token_id=66)]),
            dict(writes=[dict(output_position=31,token_id=67)]),
            dict(writes=[dict(output_position=40,token_id=68)])])
        ids,p=m.replay(Backend(),row,2)
        self.assertEqual(p,1);self.assertEqual(len(ids),33)
        self.assertEqual(ids[1],66);self.assertEqual(ids[3],65)
        self.assertEqual(ids[32],-1)
        self.assertNotIn(67,ids);self.assertNotIn(68,ids)

    def test_span_event_uses_actual_commit_order(self):
        text='x + y = 6 + 5 = 11. For B: We already know x + y = 11'
        positions=list(range(len(text)));positions[-1],positions[-2]=positions[-2],positions[-1]
        row=dict(text=text,p=dict(x=6,y=5),output_token_ids=[ord(c) for c in text],
                 steps=[dict(step=j,writes=[dict(output_position=i,token_id=ord(text[i]))]) for j,i in enumerate(positions)])
        cp,rp,s=m.spans(Tokenizer(),row)
        self.assertEqual(''.join(text[i] for i in cp),'11')
        self.assertEqual(''.join(text[i] for i in rp),'11')
        self.assertEqual(s,len(text)-2)

    def test_source_paraphrases(self):
        for text in ('x + y = 10. For B: 2*(5 + 5) = 2*10 = 20.',
                     'x + y = 5 + 5 = 10. For B: First, x + y is 10, so 2*10 = 20.'):
            row=dict(text=text,p=dict(x=5,y=5),output_token_ids=[ord(c) for c in text],
                     steps=[dict(step=j,writes=[dict(output_position=j,token_id=ord(t))]) for j,t in enumerate(text)])
            cp,rp,s=m.spans(Tokenizer(),row)
            self.assertEqual(''.join(text[i] for i in cp),'10')
            self.assertEqual(''.join(text[i] for i in rp),'10')
            self.assertGreater(rp[0],text.index('For B:'))

if __name__=='__main__':unittest.main()
