"""Original-task ARB/RAB patching and directed attention edge ablations."""
from . import precommit_v25 as v25

PROTOCOL='precommit-v2.6'
LAYOUTS=('ARB','RAB')
KINDS=v25.ALIAS_KINDS
MODES=v25.MODES
POLICIES=('confidence','B_first')
ATTENTION_MODES=('native','sham','tail_to_original','tail_to_answers',
                 'original_to_tail','tail_to_original_early','tail_to_original_late')


def dataset(seed=123):
    return dict(v25.dataset(seed),protocol=PROTOCOL)


class CanvasBuilder(v25.CanvasBuilder):
    def alias(self,item,parent,layout='ARB'):
        if layout not in LAYOUTS:raise ValueError(layout)
        # Keep the v2.5 prompt exactly; only rearrange the supplied output slots.
        cc=self.build(v25.redundant_prompt(item,parent,True),layout,2,'mask')
        return dict(cc,parent=parent,requested='ABR')


def schedule(kind):
    jobs=v25.alias_schedule(kind)
    if kind=='B_only':
        jobs += [dict(site=s,mode='L24_31',layers=MODES['L24_31']) for s in ('A','R')]
    return jobs


def attention_conditions():
    for layout in ('AB','BA'):
        for quantity in ('mask_0','mask_2','mask_32','eos_32'):
            modes=ATTENTION_MODES[:2] if quantity=='mask_0' else (ATTENTION_MODES[:3] if quantity=='mask_2' else ATTENTION_MODES)
            for mode in modes:yield layout,quantity,mode


def attention_plan(canvas,mode,num_layers=32):
    """Information key/value -> query: block [receiver query, sender key]."""
    if mode not in ATTENTION_MODES:raise ValueError(mode)
    n=len(canvas['ids']);tail=canvas['unused'];original=[i for i in range(n) if i not in tail]
    answers=sorted(p for ps in canvas['spans'].values() for p in ps)
    queries,keys,layers=[],[],[]
    if mode not in ('native','sham'):
        if not tail:raise ValueError('No tail edges to block')
        queries,keys=(tail,original) if mode=='original_to_tail' else (answers if mode=='tail_to_answers' else original,tail)
        layers=list(range(1,num_layers+1))
        if mode.endswith('_early'):layers=layers[:num_layers//2]
        if mode.endswith('_late'):layers=layers[num_layers//2:]
    if set(queries)&set(keys):raise ValueError('This protocol never blocks a self edge')
    return dict(mode=mode,length=n,query_positions=queries,key_positions=keys,layers=layers,
                blocked_edges_per_layer=len(queries)*len(keys),heads='all',
                direction='key/value sender -> query receiver',hook_all_layers=mode!='native')


class AttentionEdges:
    """Keep SDPA; inject additive -inf before softmax, with a zero-bias sham.

    Every non-native condition passes a bias to every layer, including layers
    outside the selected window. Thus edge effects are compared with sham,
    while native-vs-sham separately measures numerical backend differences.
    """
    def __init__(self,layers,canvas,mode):
        self.layers=list(layers);self.plan=attention_plan(canvas,mode,len(self.layers))
        self.handles=[];self.calls=[0]*len(self.layers);self.zero=self.blocked=None

    def __enter__(self):
        if not self.plan['hook_all_layers']:return self
        try:
            for index,layer in enumerate(self.layers):
                def hook(module,args,kwargs,index=index):
                    if len(args)>1:raise ValueError('Unexpected positional attention mask')
                    h=args[0] if args else kwargs['hidden_states']
                    if h.shape[:2]!=(1,self.plan['length']):raise ValueError('Unexpected attention shape')
                    if kwargs.get('attention_mask') is not None:raise ValueError('Expected native full attention')
                    if self.zero is None:
                        import torch
                        n=self.plan['length'];self.zero=h.new_zeros((1,1,n,n));self.blocked=self.zero.clone()
                        q=torch.tensor(self.plan['query_positions'],device=h.device,dtype=torch.long)
                        k=torch.tensor(self.plan['key_positions'],device=h.device,dtype=torch.long)
                        if q.numel() and k.numel():self.blocked[:,:,q[:,None],k[None,:]]=-torch.inf
                        if torch.isneginf(self.blocked).sum().item()!=self.plan['blocked_edges_per_layer']:
                            raise AssertionError('Incorrect edge orientation/count')
                    self.calls[index]+=1
                    return args,dict(kwargs,attention_mask=self.blocked if index+1 in self.plan['layers'] else self.zero)
                self.handles.append(layer.self_attn.register_forward_pre_hook(hook,with_kwargs=True))
        except Exception:
            self.__exit__(None,None,None);raise
        return self

    def __exit__(self,*args):
        for handle in self.handles:handle.remove()
        self.handles=[]

    def audit(self,forwards):
        expected=forwards if self.plan['hook_all_layers'] else 0
        if self.calls!=[expected]*len(self.layers):raise AssertionError(('Attention hooks not called',self.calls,expected))
        return dict(self.plan,calls_per_layer=self.calls.copy())


def specification(seed=123):
    patch_count=sum(len(schedule(kind))*(1 if kind=='self' else 2) for kind in KINDS)*4
    return dict(protocol=PROTOCOL,seed=seed,groups=30,layouts=list(LAYOUTS),parents=['A','R'],kinds=list(KINDS),
        policies=list(POLICIES),layer_modes=MODES,attention_conditions=list(attention_conditions()),
        scope='Original inspected calibration30; all conditions exploratory; no correctness filtering',
        prompt='Exact v2.5 numeric prompt, definition order A,R,B; requested A,B,R; only scaffold ARB/RAB changes',
        primary='Within fixed layout and fixed site: (change-same) when B depends on site minus when B does not',
        attention='Negative infinity at selected [receiver query,sender key]; all heads; softmax renormalizes remaining edges',
        attention_baseline='All-zero attention bias sham; native baseline separately audits kernel/numerical changes',
        space='Original AB/BA; frozen extra tokens; four commits; all-layer, 1..16, 17..32 windows fixed before inference',
        planned_per_group=dict(alias_behavior=8,alias_contexts=60,alias_patch=patch_count,
                               attention_behavior=len(list(attention_conditions()))*2))
