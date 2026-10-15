"""Secondary diagnostic: candidate gains vs loss of Target probability; no new inference."""
import argparse
import json
import math
from pathlib import Path
import statistics as st
from analyze_precommit_v22 import lines, estimate, grouped


def main():
    ap=argparse.ArgumentParser();ap.add_argument('runs',type=Path);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();out={}
    if a.out.exists():raise FileExistsError(a.out)
    for split in ('calibration','discovery'):
        root=a.runs/'illada-v22'/split;cx={r['context_id']:r for r in lines(root/'contexts.jsonl')}
        rr=[r for r in lines(root/'patch.jsonl') if r['relation']=='downstream' and r['mode']=='single' and r['layers']==[31] and r['site']=='answer_A']
        out[split]={}
        for r in rr:
            c=cx[r['context_id']];before=dict(zip(c['prediction_positions'],c['baseline']));after=dict(zip(c['prediction_positions'],r['predictions']))
            def lp(by,ids):return st.mean(math.log(max(by[p]['probabilities'][str(t)],1e-38)) for p,t in zip(c['Target']['spans']['B'],ids))
            r['candidate_logp_change']=lp(after,c['B_candidate_token_ids'])-lp(before,c['B_candidate_token_ids'])
            r['target_logp_change']=lp(after,c['B_target_token_ids'])-lp(before,c['B_target_token_ids'])
            assert abs(r['candidate_logp_change']-r['target_logp_change']-r['delta'])<1e-6
            r['hit_before']=c['baseline_values']['B']==c['alternative']['B'];r['hit_after']=r['values']['B']==c['alternative']['B']
        for kind in sorted({r['source_kind'] for r in rr}):
            rs=[r for r in rr if r['source_kind']==kind]
            out[split][kind]=dict(n=len(rs),before=sum(r['hit_before'] for r in rs),after=sum(r['hit_after'] for r in rs),
                new=sum(not r['hit_before'] and r['hit_after'] for r in rs),lost=sum(r['hit_before'] and not r['hit_after'] for r in rs),
                candidate_logp_change=estimate(grouped(rs,lambda r:r['candidate_logp_change'])),
                target_logp_change=estimate(grouped(rs,lambda r:r['target_logp_change'])))
    a.out.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
