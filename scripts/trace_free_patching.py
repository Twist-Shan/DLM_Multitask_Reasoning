"""Audit query probabilities versus actual confidence-scheduled commitments."""
import json
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling.backends.hf import HFBackend
from dlm_coupling.shared_causal import ResidualPatch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/free-reasoning-patching')
    run=parser.parse_args().run
    rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
    meta=json.loads((run/'manifest.json').read_text());assert meta['status']=='complete'
    b=HFBackend(meta['model'],load_model=True);b._torch.manual_seed(123)
    with (run/'commitment_trace.jsonl').open('x') as f:
        for ev in [r for r in rows if r['kind']=='event']:
            case=ev['case'];selection=next(r for r in rows if r['kind']=='selection' and r['case']==case)['selected'];l=selection['layer']-1
            source=next(r for r in rows if r['kind']=='source_event' and r['case']==case and r['source_kind']=='diff_C_same_answers')
            with ResidualPatch(b._model.model.layers[l],source['common_positions']) as cap:
                b.predict(source['state'],[source['common_positions'][0]],[[]])
            values=cap.captured;query=ev['reuse_positions'][-1];plen=ev['prompt_tokens']
            probes=[int(t) for t in ev['baseline']['probabilities']]
            for condition in ('baseline','activation_single','activation_block'):
                rr=next(r for r in rows if r['kind']=='rollout' and r['case']==case and r['condition']==condition)
                state=ev['state'].copy()
                for step,w in enumerate(rr['writes']):
                    assert len(state)>plen+w['output_position']
                    missing=[i for i in range(plen,len(state)) if state[i]==b.mask_id]
                    if w['patched']:
                        with ResidualPatch(b._model.model.layers[l],ev['common_positions'],values):
                            pp=b.predict(state,missing,[probes if p==query else [] for p in missing])
                    else:pp=b.predict(state,missing,[probes if p==query else [] for p in missing])
                    j=max(range(len(missing)),key=lambda i:(pp[i]['confidence'],-missing[i]))
                    pos=missing[j];pred=pp[j];q=pp[missing.index(query)]
                    assert pos==plen+w['output_position'] and pred['token_id']==w['token_id'],'Trace mismatch'
                    record=dict(case=case,condition=condition,step=step,layer=l+1,query=q,selected_position=pos-plen,selected=pred,query_committed=pos==query,patched=w['patched'])
                    f.write(json.dumps(record)+'\n');f.flush()
                    print('TRACE',case,condition,step,'query',q['token_text'],q['confidence'],'selected',pos-plen,pred['token_text'],pred['confidence'],flush=True)
                    state[pos]=pred['token_id']
                    if pos==query:break
    print('TRACE COMPLETE',flush=True)

if __name__=='__main__':main()
