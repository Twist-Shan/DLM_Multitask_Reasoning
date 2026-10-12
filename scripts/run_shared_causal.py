"""Small controlled-layout iLLaDA pilot. All layer scans are exploratory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dlm_coupling.backends.hf import HFBackend
from dlm_coupling.shared_causal import examples, answers, canvas, fill, read_values, ResidualPatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()
    if args.n < 1: raise ValueError('n must be positive')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', args=vars(args),
                    scope='fixed supplied answer layout; oracle C conditions explicitly labelled; one-forward residual patches; exploratory, no confirmatory claim',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'src/dlm_coupling/shared_causal.py').read_bytes()).hexdigest())
    def save(): (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save()
    backend = HFBackend(dict(family='illada', model_id='GSAI-ML/iLLaDA-8B-Instruct',
        revision='5769f049224b2cff61106ebce0429ede5deeb434', trust_remote_code=True,
        dtype='bfloat16', device='cuda'), load_model=True)
    backend._torch.manual_seed(args.seed); backend._torch.cuda.manual_seed_all(args.seed)
    candidates = [(name, module) for name,module in backend._model.named_modules()
                  if name in ('model.layers', 'model.transformer.blocks')]
    if len(candidates) != 1:
        raise RuntimeError('Unrecognized iLLaDA block structure: '+str([n for n,_ in backend._model.named_modules()][:80]))
    name, blocks = candidates[0]
    layers = sorted({len(blocks)//4-1, len(blocks)//2-1, 3*len(blocks)//4-1, len(blocks)-2})
    manifest.update(backend=backend.metadata, module=name, layers=layers, cases=examples(args.n,args.seed)); save()
    with (out/'results.jsonl').open('w', encoding='utf-8') as f:
        def emit(row):
            f.write(json.dumps(row)+'\n'); f.flush()
            print(json.dumps({k:v for k,v in row.items() if k not in ('ids','writes','prompt')}), flush=True)
        for case in manifest['cases']:
            base=case['base']; c=answers(base)['C']
            # C generation is scored, with A/B masks present but not scheduled yet.
            ids,spans,_,prompt=canvas(backend,base)
            gen,w=fill(backend,ids,spans['C'])
            end,w2=fill(backend,gen,spans['A']+spans['B'])
            emit(dict(case=case['id'],condition='generated_C',result=read_values(backend,end,spans),ids=end,writes=w+w2,prompt=prompt,gold=case['gold']))
            for kind in ('donor','same'):
                donor,ds,_,dp=canvas(backend,case[kind])
                donor,w=fill(backend,donor,ds['C'])
                donor,w2=fill(backend,donor,ds['A']+ds['B'])
                emit(dict(case=case['id'],condition=kind+'_generated_C',result=read_values(backend,donor,ds),ids=donor,writes=w+w2,prompt=dp,gold=answers(case[kind])))
            for supplied in (c,c+1):
                state,ss,_,_=canvas(backend,base,supplied)
                end,w=fill(backend,state,ss['A']+ss['B'])
                emit(dict(case=case['id'],condition='oracle_C' if supplied==c else 'counterfactual_C',result=read_values(backend,end,ss),expected=answers(base,supplied),ids=end,writes=w))
            for stage in ('C_masked','C_visible_oracle'):
                state,ss,_,_=canvas(backend,base,None if stage=='C_masked' else c)
                positions=ss['A']+ss['B']
                control,w=fill(backend,state,positions)
                emit(dict(case=case['id'],condition='patch_baseline',stage=stage,result=read_values(backend,control,ss),ids=control,writes=w))
                for layer in layers:
                    for kind,p in [('self',base),('changed_C',case['donor']),('same_C',case['same']),('unrelated_position',case['donor'])]:
                        donor,ds,_,_=canvas(backend,p,None if stage=='C_masked' else answers(p)['C'])
                        if len(donor)!=len(state) or ds!=ss:
                            raise ValueError('Source/recipient positions not aligned')
                        site=ss['C'] if kind!='unrelated_position' else list(range(ss['C'][0]-len(ss['C']),ss['C'][0]))
                        with ResidualPatch(blocks[layer],site) as capture:
                            backend.predict(donor,ds['A']+ds['B'],[[] for _ in positions])
                        with_patch=ResidualPatch(blocks[layer],site,capture.captured)
                        end,w=fill(backend,state,positions,with_patch)
                        if kind=='self' and end!=control:
                            raise AssertionError('Self-patch does not reproduce baseline')
                        emit(dict(case=case['id'],condition=kind,stage=stage,layer=layer,
                            result=read_values(backend,end,ss),patch_positions=site,expected=case['hybrid'] if kind=='changed_C' else case['gold'],
                            donor_gold=answers(p),ids=end,writes=w,control_exact=end==control))
    manifest.update(status='complete',backend=backend.metadata);save()
    print('SHARED CAUSAL COMPLETE', flush=True)


if __name__ == '__main__': main()
