"""Audit frozen v2.6 records; paired inference uses the original question group."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling import precommit as v1, precommit_v25 as old, precommit_v26 as v


@lru_cache(maxsize=8)
def bootstrap_indices(n,seed):
    rng=random.Random(seed)
    return [rng.choices(range(n),k=n) for _ in range(4000)]


def estimate(pairs,seed=123):
    groups=defaultdict(list)
    for case,value in pairs:groups[case].append(float(value))
    if not groups:raise ValueError('Empty estimate')
    values=[sum(groups[k])/len(groups[k]) for k in sorted(groups)]
    n=len(values)
    draws=sorted(sum(values[i] for i in ids)/n for ids in bootstrap_indices(n,seed))
    return dict(mean=sum(values)/n,ci95=[draws[99],draws[3899]],groups=n,
                observations=sum(map(len,groups.values())))


def correct(row,event='final',field='B'):
    values=row['t0']['values'] if event=='t0' else row['values']
    return values[field]==row['expected'][field]


def indexed(rows,keys):
    result={tuple(r[k] for k in keys):r for r in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate rows: '+str(keys))
    return result


def audit_behavior(row):
    cc=row['canvas'];n=len(cc['spans'])*2
    if row['nfe']!=n or row['forwards']!=n or len(row['writes'])!=n:
        raise ValueError('Unmatched decode budget')
    if any(cc['ids'][p]!=cc['mask_id'] for ps in cc['spans'].values() for p in ps):
        raise ValueError('t0 answer already committed')
    for step in row['writes']:
        if any(step['state'][p]!=cc['ids'][p] for p in cc['unused']):raise ValueError('Tail committed')
        if [step['state'][p] for p in cc['suffix_positions']]!=cc['suffix_ids']:raise ValueError('Suffix changed')
    if row['policy']=='B_first':
        for step in row['writes'][:2]:
            if step['position'] not in cc['spans']['B']:raise ValueError('Not B first')
            if any(step['state'][p]!=cc['mask_id'] for k,ps in cc['spans'].items() if k!='B' for p in ps):
                raise ValueError('Visible parent before B')
    if 'attention_mode' in row:
        plan=v.attention_plan(cc,row['attention_mode'])
        if any(row['attention'][k]!=value for k,value in plan.items()):raise ValueError('Wrong attention edges')
        calls=5 if row['policy']=='confidence' else 9
        if row['hook_forward_count']!=calls:raise ValueError('Hook forward count')
        expected_calls=calls if plan['hook_all_layers'] else 0
        if row['attention']['calls_per_layer']!=[expected_calls]*32:raise ValueError('Hook coverage')


def behavior_cells(rows,keys):
    cells=defaultdict(list)
    for row in rows:
        audit_behavior(row)
        cells['/'.join(row[k] for k in keys)].append(row)
    return {key:dict(n=len(rr),t0={k:sum(correct(r,'t0',k) for r in rr) for k in rr[0]['expected']},
                     final={k:sum(correct(r,'final',k) for r in rr) for k in rr[0]['expected']},
                     all_final=sum(all(correct(r,'final',k) for k in r['expected']) for r in rr))
            for key,rr in cells.items()}


def analyze(folder,prior=None):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='complete':raise ValueError('Incomplete run')
    spec=json.loads((folder/'specification.json').read_text(encoding='utf-8'))
    if spec!=json.loads(json.dumps(v.specification(manifest['seed']))):raise ValueError('Specification changed')
    planned=spec['planned_per_group'];n=manifest['groups']
    rows={k:[json.loads(s) for s in (folder/(k+'.jsonl')).read_text(encoding='utf-8').splitlines()] for k in planned}
    for k,count in planned.items():
        if len(rows[k])!=count*n or manifest['outputs'][k]!=len(rows[k]):raise ValueError('Row count: '+k)
    contexts=indexed(rows['alias_contexts'],('context_id',))
    data=v.dataset(manifest['seed']);groups=[g['id'] for g in data['groups'][:n]]
    if json.loads((folder/'dataset.json').read_text(encoding='utf-8'))['groups']!=data['groups'][:n]:
        raise ValueError('Original questions changed')
    for (context_id,),c in contexts.items():
        ex=old.redundant_gold(c['p'],c['parent']);src=old.redundant_source(c['p'],c['parent'],c['kind'],c['sign'])
        if ex!=c['expected'] or src!=c['source_p'] or old.redundant_gold(src,c['parent'])!=c['source_expected']:
            raise ValueError('Source arithmetic')
        if c['kind'].startswith('comp_') and ex['B']!=c['source_expected']['B']:raise ValueError('Compensation')
        old.check_alignment(c['Target'],c['Source'])
        for cc in (c['Target'],c['Source']):
            if any(cc['ids'][p]!=cc['mask_id'] for ps in cc['spans'].values() for p in ps):raise ValueError('t0 committed')
            if [cc['ids'][p] for p in cc['suffix_positions']]!=cc['suffix_ids']:raise ValueError('Native suffix')
    max_error=null_max=0.
    patch=indexed(rows['alias_patch'],('case','parent','layout','kind','sign','site','mode'))
    expected_keys=set()
    for case in groups:
        for parent in ('A','R'):
            for layout in v.LAYOUTS:
                for kind in v.KINDS:
                    for sign in ((1,) if kind=='self' else (-1,1)):
                        for job in v.schedule(kind):expected_keys.add((case,parent,layout,kind,sign,job['site'],job['mode']))
    if set(patch)!=expected_keys:raise ValueError('Missing or extra patch conditions')
    cells=defaultdict(list)
    for r in rows['alias_patch']:
        c=contexts[r['context_id'],];cc=dict(c['Target'])
        cc['maps']={'B':{c['expected']['B']:c['B_target_token_ids'],c['alternative']['B']:c['B_candidate_token_ids']}}
        base=v1.log_ratio(cc,c['baseline'],c['positions'],'B',c['expected']['B'],c['alternative']['B'])
        post=v1.log_ratio(cc,r['predictions'],c['positions'],'B',c['expected']['B'],c['alternative']['B'])
        max_error=max(max_error,abs(base-c['baseline_log_ratio']),abs(post-r['log_ratio']),abs(post-base-r['delta']))
        expected_positions=cc['unused'] if r['site']=='unused' else cc['spans'][r['site']]
        if r['layers']!=v.MODES[r['mode']] or r['patch_positions']!=expected_positions:raise ValueError('Patch site/layers')
        if set(r['patch_positions'])&set(r['raw_query_positions']):raise ValueError('Direct B overwrite')
        if r['kind']=='self' or r['layers']==[32]:null_max=max(null_max,abs(r['delta']))
        cells['/'.join(r[k] for k in ('parent','layout','kind','site','mode'))].append((r['case'],r['delta']))
    if max_error>1e-8 or null_max>1e-6:raise ValueError('Metric/null replay failed')
    result=dict(protocol=v.PROTOCOL,status='exploratory_complete',manifest=manifest,
                audit=dict(metric_max_error=max_error,null_max_abs=null_max),
                raw_sha256={k:hashlib.sha256((folder/(k+'.jsonl')).read_bytes()).hexdigest() for k in planned},
                alias_patch_cells={k:estimate(vals) for k,vals in cells.items()})
    result['alias_behavior']=behavior_cells(rows['alias_behavior'],('parent','layout','policy'))
    result['attention_behavior']=behavior_cells(rows['attention_behavior'],('layout','quantity','attention_mode','policy'))
    aliases=indexed(rows['alias_behavior'],('case','parent','layout','policy'))
    attention=indexed(rows['attention_behavior'],('case','layout','quantity','attention_mode','policy'))
    if set(aliases)!={(g,p,l,pol) for g in groups for p in ('A','R') for l in v.LAYOUTS for pol in v.POLICIES}:
        raise ValueError('Alias coverage')
    if set(attention)!={(g,l,q,m,pol) for g in groups for l,q,m in v.attention_conditions() for pol in v.POLICIES}:
        raise ValueError('Attention coverage')
    # Fix BOTH layout and patched site while switching only B's declared parent.
    contrast=defaultdict(list);adjusted=defaultdict(list)
    for mode in ('L31','L24_31'):
        for layout in v.LAYOUTS:
            for site in ('A','R'):
                other='R' if site=='A' else 'A'
                distance='near' if layout.index(site)==1 else 'far'
                for case in groups:
                    for sign in (-1,1):
                        def score(parent,prefix):return patch[case,parent,layout,prefix+'_'+site,sign,site,mode]['delta']
                        yes=score(site,'change')-score(site,'same')
                        no=score(other,'change')-score(other,'same')
                        for key in (f'{layout}/{site}/{mode}',f'all/{distance}/{mode}',f'all/all/{mode}'):
                            contrast[key].append((case,yes-no))
                        for role,value in (('relevant',yes),('irrelevant',no)):
                            adjusted[f'{layout}/{site}/{mode}/{role}'].append((case,value))
    result['alias_dependency_contrast']={k:estimate(vals) for k,vals in contrast.items()}
    result['alias_adjusted_effect']={k:estimate(vals) for k,vals in adjusted.items()}
    # Position comparison retains the same name and dependency role; source asymmetry remains documented.
    result['alias_near_minus_far']={}
    for mode in ('L31','L24_31'):
        vals=[]
        for case in groups:
            for site in ('A','R'):
                other='R' if site=='A' else 'A'
                for sign in (-1,1):
                    def interaction(layout):
                        def get(parent,prefix):return patch[case,parent,layout,prefix+'_'+site,sign,site,mode]['delta']
                        return get(site,'change')-get(site,'same')-get(other,'change')+get(other,'same')
                    near='RAB' if site=='A' else 'ARB';far='ARB' if site=='A' else 'RAB'
                    vals.append((case,interaction(near)-interaction(far)))
        result['alias_near_minus_far'][mode]=estimate(vals)
    result['attention_contrasts']={}
    for layout,quantity,mode in v.attention_conditions():
        if mode=='native':continue
        reference='native' if mode=='sham' else 'sham'
        for policy in v.POLICIES:
            for event in ('t0','final'):
                for field in ('A','B'):
                    vals=[]
                    for case in groups:
                        a=attention[case,layout,quantity,mode,policy];b=attention[case,layout,quantity,reference,policy]
                        if a['canvas']!=b['canvas']:raise ValueError('Attention changed token inputs')
                        vals.append((case,int(correct(a,event,field))-int(correct(b,event,field))))
                    key='/'.join((layout,quantity,mode,policy,event,field))
                    result['attention_contrasts'][key]=estimate(vals)
    # Physical filler invariance is a mechanistic implementation check, not a selected result.
    maximum=0.;top_mismatch=final_mismatch=0
    for case in groups:
        for layout in ('AB','BA'):
            for policy in v.POLICIES:
                a=attention[case,layout,'mask_32','tail_to_original',policy]
                b=attention[case,layout,'eos_32','tail_to_original',policy]
                for x,y in zip(a['t0_predictions'],b['t0_predictions']):
                    maximum=max(maximum,max(abs(x['probabilities'][k]-y['probabilities'][k]) for k in x['probabilities']))
                top_mismatch+=a['t0']!=b['t0'];final_mismatch+=a['values']!=b['values']
    if maximum>1e-6 or top_mismatch or final_mismatch:raise ValueError('Full isolation failed filler invariance')
    result['audit'].update(full_isolation_probe_max_difference=maximum,
                          full_isolation_t0_mismatches=top_mismatch,full_isolation_final_mismatches=final_mismatch)
    if prior:
        prior_rows=[json.loads(s) for s in (prior/'core_behavior.jsonl').read_text(encoding='utf-8').splitlines()]
        index=indexed(prior_rows,('case','relation','layout','quantity','policy'))
        comparisons=final_diff=t0_diff=canvas_diff=0
        for r in rows['attention_behavior']:
            if r['attention_mode']!='native':continue
            old_row=index[r['case'],'downstream',r['layout'],r['quantity'],r['policy']]
            comparisons+=1;canvas_diff+=r['canvas']!=old_row['canvas']
            t0_diff+=r['t0']!=old_row['t0'];final_diff+=r['values']!=old_row['values']
        result['v25_replay']=dict(comparisons=comparisons,canvas_mismatches=canvas_diff,
                                  t0_mismatches=t0_diff,final_mismatches=final_diff)
        if canvas_diff:raise ValueError('Native v2.5 replay input changed')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder',type=Path);ap.add_argument('--prior',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    result=analyze(args.folder,args.prior);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('audit','alias_dependency_contrast','alias_near_minus_far')},indent=2))


if __name__=='__main__':main()
