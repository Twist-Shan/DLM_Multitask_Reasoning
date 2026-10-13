import unittest
from dlm_coupling.implicit_sharing import data, gold, sources, prompt, canvas, rollout, expected_cf, RELATIONS

class Tokenizer:
    def encode(self,s,**kw):return list(map(ord,s))
    def decode(self,ids,**kw):return ''.join(map(chr,ids))
    def apply_chat_template(self,messages,**kw):return '<user>'+messages[0]['content']+'</user><assistant>'

class Fake:
    tokenizer=Tokenizer(); mask_id=5
    def predict(self,ids,positions,probes):
        return [dict(token_id=ord('2'),token_text='2',confidence=p/10000) for p in positions]

class Tests(unittest.TestCase):
    def test_sources_and_counterfactuals(self):
        groups=data();self.assertEqual(len(groups),24);self.assertEqual(groups,data())
        seen=set()
        for c in groups:
            p=c['p'];current=set()
            for rel in RELATIONS:
                vv=sources(p,rel)
                self.assertEqual(gold(p,rel),gold(vv['diff_C_same_answers'],rel))
                self.assertEqual(gold(p,rel),gold(vv['same_value'],rel))
                for q in vv.values():
                    self.assertTrue(all(0<int(v)<10 for v in q.values()))
                    self.assertTrue(all(10<=v<=99 for v in gold(q,rel).values()))
                    current.add(tuple(sorted(q.items())))
                expected=expected_cf(p,rel,'diff_C_same_answers')
                self.assertEqual(expected['A'],gold(p,rel)['A']+1)
                self.assertEqual(expected['B'],gold(p,rel)['B']+(0 if rel=='independent' else 2))
            self.assertFalse(current&seen);seen.update(current)
    def test_no_intermediate_or_extra_request(self):
        p=data()[0]['p']
        for rel in RELATIONS:
            text=prompt(p,rel,'A')
            self.assertIn('Requested outputs: A.',text)
            for key in 'CUV':self.assertNotIn(key+' =',text)
            cc=canvas(Fake(),p,rel,'AB')
            self.assertEqual(set(cc['spans']),set('AB'))
            for q in sources(p,rel).values():
                src=canvas(Fake(),q,rel,'AB')
                self.assertEqual(src['sites'],cc['sites'])
                self.assertEqual(len(src['ids']),len(cc['ids']))
    def test_unconstrained_commit_order_and_complete_output(self):
        b=Fake();p=data()[0]['p'];cc=canvas(b,p,'shared')
        rr=rollout(b,cc)
        self.assertIn(rr['writes'][0]['position'],cc['spans']['B'])
        self.assertEqual(rr['nfe'],4);self.assertEqual(rr['values'],dict(A=22,B=22))
        for req in ('A','B'):
            r=rollout(b,canvas(b,p,'shared',req));self.assertEqual(r['nfe'],2)
    def test_revised_prompts_and_source_alignment(self):
        p=data()[0]['p']
        for style in ('compact','numeric'):
            for rel in RELATIONS:
                for req in ('AB','BA','A','B'):
                    cc=canvas(Fake(),p,rel,req,style)
                    for q in sources(p,rel).values():
                        sc=canvas(Fake(),q,rel,req,style)
                        self.assertEqual(cc['sites'],sc['sites'])
                        self.assertEqual(cc['spans'],sc['spans'])
                        self.assertEqual(len(cc['ids']),len(sc['ids']))
                    for key in 'CUV':self.assertNotIn(key+' =',cc['prompt'])
                    if style=='numeric':
                        self.assertIn(f"A = ({p['x']} + {p['y']})",cc['prompt'])

if __name__=='__main__':unittest.main()
