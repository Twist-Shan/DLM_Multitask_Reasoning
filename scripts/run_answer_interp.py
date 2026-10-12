"""Predefined late-output intervention screen, with exact continuation controls."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling.backends.hf import HFBackend
from dlm_coupling.answer_intervention import answer_events,choose_fork,replay_state,continue_fork


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--per-relation',type=int,default=3);args=ap.parse_args()
    folder=Path(args.run);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    base=json.loads((folder/'manifest.json').read_text());assert base['status']=='complete'
    rows=[json.loads(l) for l in (folder/'results.jsonl').read_text().splitlines()]
    backend=HFBackend(base['model_config'],load_model=True)
    seed=base['args'].get('seed',123)
    backend._torch.manual_seed(seed);backend._torch.cuda.manual_seed_all(seed)
    manifest={'status':'running','args':vars(args),'seed':seed,'base_results_sha256':hashlib.sha256((folder/'results.jsonl').read_bytes()).hexdigest(),
              'source_digest':hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src').rglob('*.py')))).hexdigest(),
              'selection':'first eligible distinct pairs per relation in stored order; AB before BA; no correctness filtering',
              'scope':'late final-answer token perturbation in one block; target digits all masked; no hidden-state intervention'}
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    save();counts=Counter();seen=set();reasons=Counter();nfe=0;events=[]
    with (out/'events.jsonl').open('w') as ef,(out/'forks.jsonl').open('w') as ff:
        for row in rows:
            if len(row['order'])!=2: continue
            ev=answer_events(backend.tokenizer,row)
            event={'pair_id':row['pair_id'],'relation':row['relation'],'order':row['order'],**ev}
            options=[]
            for a,b in [('A','B'),('B','A')]:
                f,reason=choose_fork(backend.tokenizer,row,ev,a,b)
                if f: options.append(f)
                else: reasons[reason]+=1
            event['eligible_directions']=[f['source']+'->'+f['target'] for f in options]
            ef.write(json.dumps(event)+'\n');ef.flush();events.append(event)
            if not options or row['pair_id'] in seen or counts[row['relation']]>=args.per_relation: continue
            fork=options[0];probes=[]
            for changed in (False,True):
                prefix,canvas=replay_state(backend,row,fork,changed)
                positions=[len(prefix)+p for p in fork['target_positions']]
                values=[[row['trace']['output_token_ids'][p]] for p in fork['target_positions']]
                pred=backend.predict(prefix+canvas,positions,values);nfe+=1
                probs=[p['probabilities'][str(v[0])] for p,v in zip(pred,values)]
                probes.append({'baseline_digit_probabilities':probs,'sum_logprob':sum(math.log(max(p,1e-45)) for p in probs)})
            control=continue_fork(backend,row,fork,False);nfe+=control['nfe']
            if control['output_token_ids']!=row['trace']['output_token_ids']:
                manifest.update(status='failed',error='Control does not reproduce baseline',pair_id=row['pair_id']);save()
                raise AssertionError('Exact continuation control failed')
            changed=continue_fork(backend,row,fork,True);nfe+=changed['nfe']
            record={'pair_id':row['pair_id'],'relation':row['relation'],'order':row['order'],'fork':fork,
                    'baseline_answers':row['answers'],'gold':row['gold'],'probes':probes,
                    'control':control,'changed':changed,'control_exact':True,
                    'target_changed':None if changed['parse_error'] else changed['answers'].get(fork['target'])!=row['answers'][fork['target']],
                    'delta_sum_logprob':probes[1]['sum_logprob']-probes[0]['sum_logprob']}
            ff.write(json.dumps(record)+'\n');ff.flush();seen.add(row['pair_id']);counts[row['relation']]+=1
            print('FORK',row['relation'],row['pair_id'],fork['source']+'->'+fork['target'],'target_changed',record['target_changed'],
                  'parse_error',changed['parse_error'],'delta_logprob',round(record['delta_sum_logprob'],4),flush=True)
    manifest.update(status='complete',forks_by_relation=dict(counts),ineligible_direction_counts=dict(reasons),
                    experimental_nfe=nfe,events=len(events),backend=backend.metadata)
    save();print('INTERP COMPLETE',json.dumps(manifest['forks_by_relation']),flush=True)

if __name__=='__main__': main()
