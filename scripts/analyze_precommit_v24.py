"""Read-only audit and paired, group-clustered summaries for v2.4."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling import precommit as v1,precommit_v24 as v


def mean(values):return statistics.mean(values)


def estimate(pairs,seed=123):
    groups=defaultdict(list)
    for case,value in pairs:groups[case].append(float(value))
    if not groups:raise ValueError('Empty estimate')
    values=[mean(groups[k]) for k in sorted(groups)]
    rng=random.Random(seed)
    draws=sorted(mean(rng.choices(values,k=len(values))) for _ in range(4000))
    return dict(mean=mean(values),ci95=[draws[99],draws[3899]],groups=len(values),
                observations=sum(map(len,groups.values())))


def analyze(folder):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='complete':raise ValueError('Incomplete run; do not summarize as completed')
    names=('behavior','contexts','patch','rollout','space','space_contexts','space_patch')
    rows={name:[json.loads(s) for s in (folder/(name+'.jsonl')).read_text(encoding='utf-8').splitlines()] for name in names}
    for name,rr in rows.items():
        if len(rr)!=manifest['outputs'][name]:raise ValueError('Count mismatch: '+name)
    contexts={r['context_id']:r for r in rows['contexts']+rows['space_contexts']}
    if len(contexts)!=len(rows['contexts'])+len(rows['space_contexts']):raise ValueError('Duplicate context')
    planned=dict(behavior=8,contexts=60,patch=196,rollout=16,
                 space=22,space_contexts=40,space_patch=80)
    if any(len(rows[k])!=n*manifest['groups'] for k,n in planned.items()):
        raise ValueError('Protocol count mismatch')
    for c in contexts.values():
        if v.gold(c['p'],c['parent'])!=c['expected'] or v.gold(c['source_p'],c['parent'])!=c['source_expected']:
            raise ValueError('Recorded arithmetic does not match operands')
        if v.source(c['p'],c['parent'],c['kind'],c['sign'])!=c['source_p']:
            raise ValueError('Source construction mismatch')
        v.check_alignment(c['Target'],c['Source'])
        for cc in (c['Target'],c['Source']):
            if any(cc['ids'][p]!=cc['mask_id'] for ps in cc['spans'].values() for p in ps):
                raise ValueError('t=0 contains a committed answer')
            if [cc['ids'][p] for p in cc['suffix_positions']]!=cc['suffix_ids']:
                raise ValueError('Native suffix altered')
    maximum_error=0.;null_max=0.
    for r in rows['patch']+rows['space_patch']:
        c=contexts[r['context_id']];cc=dict(c['Target'])
        # Replay with exact token IDs from the audited full-context maps.
        if 'B_target_token_ids' not in c:raise ValueError('Missing exact metric token IDs')
        cc['maps']={'B':{c['expected']['B']:c['B_target_token_ids'],c['alternative']['B']:c['B_candidate_token_ids']}}
        base=v1.log_ratio(cc,c['baseline'],c['positions'],'B',c['expected']['B'],c['alternative']['B'])
        score=v1.log_ratio(cc,r['predictions'],c['positions'],'B',c['expected']['B'],c['alternative']['B'])
        err=max(abs(base-c['baseline_log_ratio']),abs(score-r['log_ratio']),abs(score-base-r['delta']))
        maximum_error=max(maximum_error,err)
        if err>1e-8:raise ValueError('Metric replay mismatch')
        if r['kind']=='self' or r['layers']==[32]:null_max=max(null_max,abs(r['delta']))
        if r['kind'].startswith('comp_') and c['source_expected']['B']!=c['expected']['B']:
            raise ValueError('Compensation did not hold B fixed')
    if null_max>1e-6:raise ValueError('Null failed')
    result=dict(protocol=v.PROTOCOL,status='exploratory_complete',manifest=manifest,
        raw_sha256={name:hashlib.sha256((folder/(name+'.jsonl')).read_bytes()).hexdigest() for name in names},
        audit=dict(metric_replay_max_error=maximum_error,self_final_layer_max_abs=null_max,rows={k:len(x) for k,x in rows.items()}))
    baseline={(r['case'],r['parent'],r['layout'],r['policy']):r for r in rows['behavior']}
    if len(baseline)!=len(rows['behavior']):raise ValueError('Duplicate behavior')
    result['behavior']={}
    for parent in ('A','C'):
        for layout in v.LAYOUTS:
            rr=[r for r in rows['behavior'] if r['parent']==parent and r['layout']==layout and r['policy']=='B_first']
            group=dict(n=len(rr),t0_B=sum(r['t0']['values']['B']==r['expected']['B'] for r in rr),
                       t0_A=sum(r['t0']['values']['A']==r['expected']['A'] for r in rr),
                       t0_C=sum(r['t0']['values']['C']==r['expected']['C'] for r in rr),
                       B_first=sum(r['values']['B']==r['expected']['B'] for r in rr),
                       confidence=sum(baseline[r['case'],parent,layout,'confidence']['values']['B']==r['expected']['B'] for r in rr))
            result['behavior'][parent+'/'+layout]=group
    patches={(r['case'],r['parent'],r['layout'],r['kind'],r['sign'],r['site'],r['mode']):r for r in rows['patch']}
    if len(patches)!=len(rows['patch']):raise ValueError('Duplicate patch')
    groups=sorted({r['case'] for r in rows['patch']})
    effects={}
    for mode in ('L31','L24_31'):
        for side in ('all','left','right'):
            for equality in ('all','equal','unequal'):
                vals=[]
                for r in rows['patch']:
                    if r['kind']!='change_'+r['parent'] or r['site']!=r['parent'] or r['mode']!=mode:continue
                    if side!='all' and r['parent_side']!=side:continue
                    if equality!='all' and r['equal_AC']!=(equality=='equal'):continue
                    case,parent,layout,sign=r['case'],r['parent'],r['layout'],r['sign']
                    other='C' if parent=='A' else 'A'
                    def get(kind,site):return patches[case,parent,layout,kind,sign,site,mode]['delta']
                    vals.append((case,get('change_'+parent,parent)-get('same_'+parent,parent)
                                      -get('change_'+other,other)+get('same_'+other,other)))
                effects[f'{mode}/{side}/{equality}']=estimate(vals)
    result['dependency_selectivity']=effects
    result['dependency_selectivity_by_parent']={}
    for mode in ('L31','L24_31'):
        for parent in ('A','C'):
            vals=[];other='C' if parent=='A' else 'A'
            for case in groups:
                for layout in v.LAYOUTS:
                    for sign in (-1,1):
                        def get(prefix,site):
                            return patches[case,parent,layout,prefix+'_'+site,sign,site,mode]['delta']
                        vals.append((case,get('change',parent)-get('same',parent)
                                          -get('change',other)+get('same',other)))
            result['dependency_selectivity_by_parent'][mode+'/'+parent]=estimate(vals)
    result['patch_cells']={}
    for mode in v.MODES:
        for kind_prefix in ('change','same','comp'):
            for relevance in ('relevant','irrelevant'):
                rr=[r for r in rows['patch'] if r['mode']==mode and r['site'] in ('A','C')
                    and r['kind']==kind_prefix+'_'+r['site']
                    and ((r['site']==r['parent'])==(relevance=='relevant'))]
                if rr:result['patch_cells'][f'{mode}/{kind_prefix}/{relevance}']=estimate([(r['case'],r['delta']) for r in rr])
    result['routing_controls']={}
    for mode in ('L31','L24_31'):
        for prefix in ('change','self','B_only'):
            for site in ('relevant','irrelevant','unused'):
                rr=[r for r in rows['patch'] if r['mode']==mode
                    and (r['kind'].startswith('change_') if prefix=='change' else r['kind']==prefix)
                    and (r['site']=='unused' if site=='unused' else
                         r['site'] in ('A','C') and ((r['site']==r['parent'])==(site=='relevant')))]
                if rr:result['routing_controls'][mode+'/'+prefix+'/'+site]=estimate([(r['case'],r['delta']) for r in rr])
    result['spatial_contrasts']={}
    for mode in ('L31','L24_31'):
        vals=[]
        for case in groups:
            for parent in ('A','C'):
                left='ABC' if parent=='A' else 'CBA';right='CBA' if parent=='A' else 'ABC'
                for sign in (-1,1):
                    def adjusted(layout):
                        return (patches[case,parent,layout,'change_'+parent,sign,parent,mode]['delta']
                                -patches[case,parent,layout,'same_'+parent,sign,parent,mode]['delta'])
                    vals.append((case,adjusted(right)-adjusted(left)))
        result['spatial_contrasts'][mode+'/right_minus_left']=estimate(vals)
    result['rollout']={}
    for mode in ('first','persistent'):
        for side in ('all','left','right'):
            rr=[r for r in rows['rollout'] if r['patch_mode']==mode and (side=='all' or r['parent_side']==side)]
            before=[];after=[];new=lost=consistent=parent_transferred=0
            complete_before=complete_after=complete_new=complete_lost=0
            for r in rr:
                base=baseline[r['case'],r['parent'],r['layout'],'B_first']
                pre=base['values']['B']==r['candidate_B'];post=r['values']['B']==r['candidate_B']
                before.append(pre);after.append(post);new+=post and not pre;lost+=pre and not post
                original_parent=r['expected'][r['parent']]
                consistent+=r['values'][r['parent']] is not None and r['values']['B']==r['values'][r['parent']]+r['expected']['B']-original_parent
                parent_transferred+=post and r['values'][r['parent']]==original_parent+r['sign']
                alternative=dict(r['expected'],B=r['candidate_B'])
                alternative[r['parent']]+=r['sign']
                cb=base['values']==alternative;ca=r['values']==alternative
                complete_before+=cb;complete_after+=ca;complete_new+=ca and not cb;complete_lost+=cb and not ca
                for w in r['writes'][:2]:
                    if w['position'] not in r['canvas']['spans']['B']:raise ValueError('B-first failed')
                    for k in ('A','C'):
                        if any(w['state'][p]!=r['canvas']['mask_id'] for p in r['canvas']['spans'][k]):raise ValueError('Parent committed before B')
            result['rollout'][mode+'/'+side]=dict(n=len(rr),baseline_candidate=sum(before),after_candidate=sum(after),
                new=new,lost=lost,relationship_consistent=consistent,candidate_and_parent_transferred=parent_transferred,
                complete_counterfactual=dict(baseline=complete_before,after=complete_after,new=complete_new,lost=complete_lost),
                paired_gain=estimate([(r['case'],float(post)-float(pre)) for r,pre,post in zip(rr,before,after)]))
    space={(r['case'],r['parent'],r['condition']):r for r in rows['space']}
    if len(space)!=len(rows['space']):raise ValueError('Duplicate space behavior')
    result['space']={}
    for cond in v.specification()['space_quantity']+v.specification()['space_usage']:
        rr=[r for r in rows['space'] if r['condition']==cond]
        deltas=[]
        for r in rr:
            base=space[r['case'],r['parent'],'mask_0']
            deltas.append((r['case'],float(r['values']['B']==r['expected']['B'])-float(base['values']['B']==base['expected']['B'])))
            if r['nfe']!=2 or r['forwards']!=2:raise ValueError('Space altered B decoding budget')
            if r['canvas']['spans']['B']!=base['canvas']['spans']['B']:raise ValueError('B moved')
            if r['canvas']['prompt']!=base['canvas']['prompt']:raise ValueError('Space prompt changed')
            cc=r['canvas'];allowed=set(cc['spans']['B'])
            for state in [r['ids']]+[w['state'] for w in r['writes']]:
                if any(state[p]!=cc['ids'][p] for p in range(len(state)) if p not in allowed):
                    raise ValueError('Space decoder changed non-B position')
        result['space'][cond]=dict(n=len(rr),t0_B=sum(r['t0']['values']['B']==r['expected']['B'] for r in rr),
            final_B=sum(r['values']['B']==r['expected']['B'] for r in rr),
            valid_B=sum(r['values']['B'] is not None for r in rr),
            mean_canvas_tokens=mean(len(r['canvas']['ids']) for r in rr),
            mean_decode_seconds=mean(r['seconds'] for r in rr),paired_final_gain=estimate(deltas))
    result['space_semantic_contrasts']={}
    for comparison in ('unrelated_slot','matched_mask','matched_eos'):
        for event in ('t0','final'):
            vals=[]
            for case in groups:
                for parent in ('A','C'):
                    related=space[case,parent,'slot_'+parent]
                    control_name='slot_'+('C' if parent=='A' else 'A') if comparison=='unrelated_slot' else comparison
                    control=space[case,parent,control_name]
                    rc,cc=related['canvas'],control['canvas']
                    if len(rc['ids'])!=len(cc['ids']) or rc['suffix_positions']!=cc['suffix_positions']:
                        raise ValueError('Space usage geometry mismatch')
                    def correct(r):return (r['t0']['values']['B'] if event=='t0' else r['values']['B'])==r['expected']['B']
                    vals.append((case,float(correct(related))-float(correct(control))))
            result['space_semantic_contrasts'][comparison+'/'+event]=estimate(vals)
    result['space_patch']={}
    for mode in ('L31','L24_31'):
        for category in ('related_slot','unrelated_slot','matched_mask','matched_eos'):
            for prefix in ('comp','same','self'):
                rr=[]
                for r in rows['space_patch']:
                    if r['mode']!=mode or (r['kind'].split('_')[0]!=prefix):continue
                    cat=('related_slot' if r['condition']=='slot_'+r['parent'] else 'unrelated_slot') if r['condition'].startswith('slot_') else r['condition']
                    if cat==category:rr.append(r)
                result['space_patch'][mode+'/'+category+'/'+prefix]=estimate([(r['case'],r['delta']) for r in rr])
    space_patches={(r['case'],r['parent'],r['condition'],r['kind'],r['sign'],r['mode']):r for r in rows['space_patch']}
    if len(space_patches)!=len(rows['space_patch']):raise ValueError('Duplicate space patch')
    result['space_patch_contrasts']={}
    for mode in ('L31','L24_31'):
        for control in ('unrelated_slot','matched_mask','matched_eos'):
            vals=[]
            for case in groups:
                for parent in ('A','C'):
                    related='slot_'+parent
                    other='slot_'+('C' if parent=='A' else 'A') if control=='unrelated_slot' else control
                    for sign in (-1,1):
                        def get(cond,kind):return space_patches[case,parent,cond,kind+'_'+parent,sign,mode]['delta']
                        vals.append((case,get(related,'comp')-get(related,'same')-get(other,'comp')+get(other,'same')))
            result['space_patch_contrasts'][mode+'/'+control]=estimate(vals)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('folder',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    result=analyze(args.folder)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('audit','behavior','dependency_selectivity','rollout','space','space_semantic_contrasts')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
