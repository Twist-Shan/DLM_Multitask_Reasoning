"""Summarize paired discovery effects; do not turn layer search into confirmation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def interval(values, seed=123, repeats=2000):
    if not values:return None
    rng=random.Random(seed)
    means=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(repeats))
    return [means[int(.025*repeats)],means[min(repeats-1,int(.975*repeats))]]


def summarize(rows):
    grouped=defaultdict(dict)
    for r in rows:
        if r['kind'] not in ('scan','copy_scan'):continue
        key=(r['case'],r['relation'],r['requested'],r['direction'],r['source_kind'],r['layer'])
        if r['site'] in grouped[key]:raise ValueError('Duplicate case/site measurement')
        grouped[key][r['site']]=r
    cells=defaultdict(list);by_group=defaultdict(dict)
    for key,sites in grouped.items():
        case,rel,req,direction,kind,layer=key
        if set(sites)!={'answer_'+direction[0],'unused'}:
            raise ValueError('Incomplete paired sites')
        answer,unused=sites['answer_'+direction[0]],sites['unused']
        excess=answer['delta']-unused['delta']
        cells[(rel,req,direction,kind,layer)].append((case,excess,answer.get('eligible',False)))
        if rel!='copy':by_group[(req,direction,kind,layer)][case,rel]=excess
    result=[]
    for key,values in sorted(cells.items()):
        numbers=[v for _,v,_ in values]
        eligible=[v for _,v,e in values if e]
        result.append(dict(relation=key[0],requested=key[1],direction=key[2],source_kind=key[3],layer=key[4],
            groups=len(numbers),mean_excess_delta=statistics.mean(numbers),
            descriptive_paired_bootstrap95=interval(numbers),eligible_groups=len(eligible),
            eligible_mean_excess_delta=statistics.mean(eligible) if eligible else None,
            eligible_descriptive_bootstrap95=interval(eligible)))
    contrasts=[]
    for key,values in sorted(by_group.items()):
        cases=sorted({c for c,r in values})
        for relation in ('shared','downstream'):
            pairs=[values[c,relation]-values[c,'independent'] for c in cases
                   if (c,relation) in values and (c,'independent') in values]
            contrasts.append(dict(requested=key[0],direction=key[1],source_kind=key[2],layer=key[3],
                contrast=relation+' minus independent',groups=len(pairs),
                mean=statistics.mean(pairs) if pairs else None,
                descriptive_paired_bootstrap95=interval(pairs)))
    rollouts=defaultdict(list)
    for r in rows:
        if r['kind']!='rollout':continue
        key=(r['relation'],r['requested'],r['direction'],r['source_kind'],r['site'],r['schedule'])
        destination=r['direction'][1]
        other=r['direction'][0]
        rollouts[key].append(dict(valid=r['valid'],
            changed=r['values'][destination]!=r['baseline_values'][destination],
            cf=r['values'][destination]==r['expected_cf'][destination],
            new_cf=(r['values'][destination]==r['expected_cf'][destination]
                    and r['baseline_values'][destination]!=r['expected_cf'][destination]),
            other_changed=r['values'][other]!=r['baseline_values'][other],
            joint_cf=r['values']==r['expected_cf']))
    rollout_summary=[dict(condition=list(key),n=len(v),**{name:sum(r[name] for r in v)
                      for name in ('valid','changed','cf','new_cf','other_changed','joint_cf')}) for key,v in sorted(rollouts.items())]
    baselines=defaultdict(list)
    for r in rows:
        if r['kind']=='baseline':
            baselines[(r['relation'],r['requested'],r['unused'],r['source_kind'])].append(r)
    baseline_summary=[dict(relation=k[0],requested=k[1],unused=k[2],source_kind=k[3],n=len(v),
        correct=sum(r['correct'] for r in v),valid=sum(r['valid'] for r in v))
        for k,v in sorted(baselines.items())]
    return dict(scope='Exploratory discovery; paired group bootstrap is descriptive, no multiplicity correction or confirmatory claim',
        metric='answer-region delta minus same-Source-vector unused-position delta',
        cells=[r for r in result if r['relation']!='copy'],
        copy_cells=[r for r in result if r['relation']=='copy'],
        relation_contrasts=contrasts,rollouts=rollout_summary,baselines=baseline_summary,
        copy_control_records=sum(r['kind']=='copy_scan' for r in rows),
        note='Do not pool copying controls, layers, opposite directions or answer orders as independent cases')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    meta=json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
    if meta['status']!='discovery_complete':raise ValueError('Require a complete run; do not silently summarize partial data')
    rows=[json.loads(line) for line in (args.run/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    result=summarize(rows);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(args.out)


if __name__=='__main__':main()
