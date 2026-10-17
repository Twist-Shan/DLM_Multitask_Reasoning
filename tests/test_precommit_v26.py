from types import SimpleNamespace
import unittest
from dlm_coupling import precommit_v25 as old,precommit_v26 as v
from test_precommit_v24 import Backend
from test_precommit import torch


class TaskTests(unittest.TestCase):
    def test_exact_old_dataset_and_prompt(self):
        data=v.dataset();self.assertEqual(data['groups'],old.dataset()['groups'])
        b=Backend();builder=v.CanvasBuilder(b);item=old.redundant(data['groups'][0]['p'])
        for layout in v.LAYOUTS:
            for parent in ('A','R'):
                cc=builder.alias(item,parent,layout)
                self.assertEqual(cc['prompt'],old.redundant_prompt(item,parent))
                self.assertEqual(list(cc['spans']),list(layout))
                self.assertGreater(min(cc['spans']['B']),max(cc['spans']['A']+cc['spans']['R']))
        self.assertEqual(builder.alias(item,'A','ARB')['spans']['B'],builder.alias(item,'R','RAB')['spans']['B'])

    def test_all_sources_align_and_compensations_hold(self):
        builder=v.CanvasBuilder(Backend())
        for g in v.dataset()['groups']:
            item=old.redundant(g['p'])
            for parent in ('A','R'):
                ex=old.redundant_gold(item,parent)
                for kind in v.KINDS:
                    for sign in (-1,1):
                        q=old.redundant_source(item,parent,kind,sign)
                        if kind.startswith('comp_'):self.assertEqual(ex['B'],old.redundant_gold(q,parent)['B'])
                        for layout in v.LAYOUTS:old.check_alignment(builder.alias(item,parent,layout),builder.alias(q,parent,layout))

    def test_schedule_contains_source_B_control_at_both_modes(self):
        self.assertEqual({r['mode'] for r in v.schedule('B_only')},{'L31','L24_31'})
        self.assertEqual(v.specification()['planned_per_group'],dict(alias_behavior=8,alias_contexts=60,alias_patch=212,attention_behavior=76))

    def test_attention_edges_are_receiver_query_sender_key(self):
        cc=dict(ids=[0]*7,unused=[4,5],spans={'A':[0,1],'B':[2,3]})
        p=v.attention_plan(cc,'tail_to_original')
        self.assertEqual(p['query_positions'],[0,1,2,3,6]);self.assertEqual(p['key_positions'],[4,5])
        q=v.attention_plan(cc,'original_to_tail')
        self.assertEqual(q['query_positions'],p['key_positions']);self.assertEqual(q['key_positions'],p['query_positions'])
        self.assertEqual(v.attention_plan(cc,'tail_to_answers')['query_positions'],[0,1,2,3])
        self.assertEqual(v.attention_plan(cc,'tail_to_original_early')['layers'],list(range(1,17)))
        self.assertEqual(v.attention_plan(cc,'tail_to_original_late')['layers'],list(range(17,33)))
        with self.assertRaises(ValueError):v.attention_plan(dict(cc,unused=[]),'tail_to_original')


@unittest.skipIf(torch is None,'torch is required for attention hook integration')
class HookTests(unittest.TestCase):
    def fixture(self):
        class Attention(torch.nn.Module):
            def forward(self,hidden_states,attention_mask=None):
                scores=hidden_states@hidden_states.transpose(-1,-2)
                if attention_mask is not None:scores=scores+attention_mask[:,0]
                self.weights=scores.softmax(-1)
                return self.weights@hidden_states
        layers=[SimpleNamespace(self_attn=Attention()) for _ in range(2)]
        cc=dict(ids=[0]*3,unused=[2],spans={'A':[0],'B':[1]})
        return layers,cc

    def test_sham_zero_mask_probabilities_and_direction(self):
        layers,cc=self.fixture();h=torch.tensor([[[1.,0.],[0.,1.],[2.,3.]]])
        native=layers[0].self_attn(h)
        with v.AttentionEdges(layers,cc,'sham') as hook:
            for layer in layers:y=layer.self_attn(h)
        hook.audit(1);self.assertTrue(torch.equal(y,native))
        with v.AttentionEdges(layers,cc,'tail_to_original') as hook:
            for layer in layers:layer.self_attn(h)
        hook.audit(1)
        weights=layers[0].self_attn.weights
        self.assertTrue(torch.equal(weights[0,:2,2],torch.zeros(2)))
        self.assertGreater(weights[0,2,0].item(),0)
        self.assertTrue(torch.allclose(weights.sum(-1),torch.ones(1,3)))

    def test_full_isolation_prevents_tail_change_reaching_original(self):
        layers,cc=self.fixture();a=torch.tensor([[[1.,0.],[0.,1.],[2.,3.]]]);b=a.clone();b[0,2]=torch.tensor([9.,-3.])
        results=[]
        for h in (a,b):
            with v.AttentionEdges(layers,cc,'tail_to_original') as hook:
                for layer in layers:h=layer.self_attn(h)+h
            hook.audit(1);results.append(h)
        self.assertTrue(torch.equal(results[0][:,:2],results[1][:,:2]))

    def test_hooks_restore_on_exception_and_validate_calls(self):
        layers,cc=self.fixture()
        with self.assertRaisesRegex(RuntimeError,'test'):
            with v.AttentionEdges(layers,cc,'tail_to_original') as hook:raise RuntimeError('test')
        self.assertTrue(all(len(layer.self_attn._forward_pre_hooks)==0 for layer in layers))
        with self.assertRaises(AssertionError):hook.audit(1)


if __name__=='__main__':unittest.main()
