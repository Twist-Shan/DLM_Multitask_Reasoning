"""Audit v2.5 and summarize paired changes on the unchanged original questions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling import precommit as v1,precommit_v2 as v2,precommit_v25 as v
from analyze_precommit_v24 import estimate


def correct(row,event='final',field='B'):
    values=row['t0']['values'] if event=='t0' else row['values']
    return values[field]==row['expected'][field]


def analyze(folder):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='complete':raise ValueError('Incomplete run')
    planned=v.specification()['planned_per_group']
    rows={k:[json.loads(s) for s in (folder/(k+'.jsonl')).read_text(encoding='utf-8').splitlines()] for k in planned}
    for k,n in planned.items():
        if len(rows[k])!=n*manifest['groups'] or manifest['outputs'][k]!=len(rows[k]):raise ValueError('Row count: '+k)
    all_contexts=rows['core_contexts']+rows['alias_contexts']
    contexts={r['context_id']:r for r in all_contexts}
    if len(contexts)!=len(all_contexts):raise ValueError('Duplicate context')
    for group in ('core','alias'):
        for c in rows[group+'_contexts']:
            if group=='core':
                expected=v2.gold(c['p'],c['relation']);q=v.core_source(c['p'],c['kind'],c['sign'])
                source_expected=v2.gold(q,c['relation'])
            else:
                expected=v.redundant_gold(c['p'],c['parent']);q=v.redundant_source(c['p'],c['parent'],c['kind'],c['sign'])
                source_expected=v.redundant_gold(q,c['parent'])
            if expected!=c['expected'] or q!=c['source_p'] or source_expected!=c['source_expected']:
                raise ValueError('Source arithmetic mismatch')
            if c['kind'].startswith('comp_') or c['kind']=='A_change_B_fixed':
                if expected['B']!=source_expected['B']:raise ValueError('Compensation failed')
            v.check_alignment(c['Target'],c['Source'])
            for cc in (c['Target'],c['Source']):
                if any(cc['ids'][p]!=cc['mask_id'] for ps in cc['spans'].values() for p in ps):raise ValueError('t0 committed')
                if [cc['ids'][p] for p in cc['suffix_positions']]!=cc['suffix_ids']:raise ValueError('Suffix altered')
    max_error=null_max=0.
    for r in rows['core_patch']+rows['alias_patch']:
        c=contexts[r['context_id']];cc=dict(c['Target'])
        cc['maps']={'B':{c['expected']['B']:c['B_target_token_ids'],c['alternative']['B']:c['B_candidate_token_ids']}}
        base=v1.log_ratio(cc,c['baseline'],c['positions'],'B',c['expected']['B'],c['alternative']['B'])
        post=v1.log_ratio(cc,r['predictions'],c['positions'],'B',c['expected']['B'],c['alternative']['B'])
        max_error=max(max_error,abs(base-c['baseline_log_ratio']),abs(post-r['log_ratio']),abs(post-base-r['delta']))
        if set(r['patch_positions'])&set(r['raw_query_positions']):raise ValueError('Direct B overwrite')
        if r['kind']=='self' or r['layers']==[32]:null_max=max(null_max,abs(r['delta']))
    if max_error>1e-8 or null_max>1e-6:raise ValueError('Metric or null mismatch')
    result=dict(protocol=v.PROTOCOL,manifest=manifest,audit=dict(metric_max_error=max_error,null_max_abs=null_max),
                raw_sha256={k:hashlib.sha256((folder/(k+'.jsonl')).read_bytes()).hexdigest() for k in planned})
    for name in ('core_behavior','alias_behavior'):
        cells=defaultdict(list)
        for row in rows[name]:
            keys=('relation','layout','quantity','policy') if name.startswith('core') else ('parent','layout','policy')
            cells['/'.join(row[k] for k in keys)].append(row)
            cc=row['canvas'];expected_n=len(cc['spans'])*2
            if row['nfe']!=expected_n or row['forwards']!=expected_n:raise ValueError('Unmatched decode budget')
            for step in row['writes']:
                if any(step['state'][p]!=cc['ids'][p] for p in cc['unused']):raise ValueError('Extra token committed')
            if row['policy']=='B_first':
                for step in row['writes'][:2]:
                    if step['position'] not in cc['spans']['B']:raise ValueError('B-first violated')
                    if any(step['state'][p]!=cc['mask_id'] for k,ps in cc['spans'].items() if k!='B' for p in ps):
                        raise ValueError('Parent committed before B')
        result[name]={key:dict(n=len(rr),t0={field:sum(correct(r,'t0',field) for r in rr) for field in rr[0]['expected']},
                              final={field:sum(correct(r,'final',field) for r in rr) for field in rr[0]['expected']},
                              all_final=sum(all(correct(r,'final',field) for field in r['expected']) for r in rr))
                      for key,rr in cells.items()}
    core={(r['case'],r['relation'],r['layout'],r['quantity'],r['policy']):r for r in rows['core_behavior']}
    aliases={(r['case'],r['parent'],r['layout'],r['policy']):r for r in rows['alias_behavior']}
    if len(core)!=len(rows['core_behavior']) or len(aliases)!=len(rows['alias_behavior']):raise ValueError('Duplicate behavior')
    groups=sorted({r['case'] for r in rows['core_behavior']})
    result['space_gain_from_original']={}
    for layout in ('AB','BA'):
        for quantity in v.QUANTITIES:
            for policy in ('confidence','B_first'):
                for event in ('t0','final'):
                    vals=[]
                    for case in groups:
                        a=core[case,'downstream',layout,quantity,policy];base=core[case,'downstream',layout,'mask_2',policy]
                        if a['canvas']['spans']!=base['canvas']['spans'] or a['canvas']['prompt']!=base['canvas']['prompt']:
                            raise ValueError('Space moved answers or prompt')
                        vals.append((case,int(correct(a,event))-int(correct(base,event))))
                    result['space_gain_from_original']['/'.join((layout,quantity,policy,event))]=estimate(vals)
    result['space_filler_contrasts']={}
    for layout in ('AB','BA'):
        for size in (2,8,32):
            for policy in ('confidence','B_first'):
                for event in ('t0','final'):
                    for reference in ('matched_mask','zero_extra'):
                        vals=[]
                        for case in groups:
                            a=core[case,'downstream',layout,f'eos_{size}',policy]
                            q=f'mask_{size}' if reference=='matched_mask' else 'mask_0'
                            b=core[case,'downstream',layout,q,policy]
                            if reference=='matched_mask' and len(a['canvas']['ids'])!=len(b['canvas']['ids']):
                                raise ValueError('EOS/mask length mismatch')
                            vals.append((case,int(correct(a,event))-int(correct(b,event))))
                        result['space_filler_contrasts']['/'.join((layout,str(size),policy,event,reference))]=estimate(vals)
    result['alias_baseline_contrasts']={}
    for parent in ('A','R'):
        for policy in ('confidence','B_first'):
            for event in ('t0','final'):
                for layout in ('AB','ABR','RBA'):
                    vals=[(case,int(correct(aliases[case,parent,layout,policy],event))-
                                int(correct(core[case,'downstream','AB','mask_2',policy],event))) for case in groups]
                    result['alias_baseline_contrasts']['/'.join((parent,layout,policy,event))]=estimate(vals)
    for name,identity_keys in (('core_patch',('case','relation','layout','kind','sign','site','mode')),
                               ('alias_patch',('case','parent','layout','kind','sign','site','mode'))):
        index={tuple(r[k] for k in identity_keys):r for r in rows[name]}
        if len(index)!=len(rows[name]):raise ValueError('Duplicate patch')
        cells=defaultdict(list)
        for r in rows[name]:
            keys=('relation','layout','kind','site','mode') if name=='core_patch' else ('parent','layout','kind','site','mode')
            cells['/'.join(r[k] for k in keys)].append((r['case'],r['delta']))
        result[name+'_cells']={key:estimate(vals) for key,vals in cells.items()}
        if name=='core_patch':
            result['core_dependency_contrast']={}
            for layout in ('AB','BA'):
                for mode in ('L31','L24_31'):
                    vals=[]
                    for case in groups:
                        for sign in (-1,1):
                            def get(rel,kind):return index[case,rel,layout,kind,sign,'A',mode]['delta']
                            vals.append((case,get('downstream','change_A')-get('downstream','same_A')
                                          -get('independent','change_A')+get('independent','same_A')))
                    result['core_dependency_contrast'][layout+'/'+mode]=estimate(vals)
        else:
            result['alias_dependency_selectivity']={}
            for mode in ('L31','L24_31'):
                for side in ('all','left','right'):
                    vals=[]
                    for case in groups:
                        for parent in ('A','R'):
                            other='R' if parent=='A' else 'A'
                            for layout in ('ABR','RBA'):
                                actual_side='left' if layout.index(parent)<layout.index('B') else 'right'
                                if side!='all' and side!=actual_side:continue
                                for sign in (-1,1):
                                    def get(prefix,site):return index[case,parent,layout,prefix+'_'+site,sign,site,mode]['delta']
                                    vals.append((case,get('change',parent)-get('same',parent)-get('change',other)+get('same',other)))
                    result['alias_dependency_selectivity'][mode+'/'+side]=estimate(vals)
    result['rollout']={}
    for layout in ('AB','BA'):
        for mode in ('first','persistent'):
            rr=[r for r in rows['core_rollout'] if r['layout']==layout and r['patch_mode']==mode]
            pre=post=full_pre=full_post=new=lost=0;gains=[]
            for r in rr:
                base=core[r['case'],'downstream',layout,'mask_2','B_first']
                before=base['values']['B']==r['candidate_B'];after=r['values']['B']==r['candidate_B']
                pre+=before;post+=after;new+=after and not before;lost+=before and not after
                candidate={k:value+r['sign'] for k,value in r['expected'].items()}
                full_pre+=base['values']==candidate;full_post+=r['values']==candidate
                gains.append((r['case'],int(after)-int(before)))
                for step in r['writes'][:2]:
                    if step['position'] not in r['canvas']['spans']['B']:raise ValueError('Patched rollout not B-first')
                    if any(step['state'][p]!=r['canvas']['mask_id'] for p in r['canvas']['spans']['A']):raise ValueError('A visible before B')
            result['rollout'][layout+'/'+mode]=dict(n=len(rr),baseline_candidate=pre,after_candidate=post,new=new,lost=lost,
                complete_counterfactual_before=full_pre,complete_counterfactual_after=full_post,paired_gain=estimate(gains))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('folder',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    result=analyze(args.folder);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('audit','core_dependency_contrast','alias_dependency_selectivity','rollout')},indent=2))


if __name__=='__main__':main()
