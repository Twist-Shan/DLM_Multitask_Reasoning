"""Independently audit and summarize private v2.2 outputs; group bootstrap seed123."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as st
import sys
import zipfile
from functools import lru_cache

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling import precommit as v1, precommit_v2 as v2, precommit_v22 as v22


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def lines(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(None)
def bootstrap_indices(n):
    rng=random.Random(123)
    return [rng.choices(range(n),k=n) for _ in range(4000)]


def estimate(values):
    if isinstance(values,dict):values=[values[k] for k in sorted(values)]
    n=len(values)
    draws=sorted(math.fsum(values[i] for i in indices)/n for indices in bootstrap_indices(n))
    return dict(mean=st.mean(values),ci=[draws[99],draws[3899]],groups=n)


def grouped(rr,fn):
    by=defaultdict(list)
    for r in rr:by[r['case']].append(float(fn(r)))
    return {k:st.mean(v) for k,v in by.items()}


def contrast(left,right):
    assert set(left)==set(right)
    return estimate({k:left[k]-right[k] for k in left})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    data=v2.dataset();spec=v22.specification()
    suite=load(args.runs/'suite_status.json');assert suite['status']=='finished'
    code=zipfile.ZipFile(args.runs/'code-v22.zip')
    hashes=json.loads(code.read('BUNDLE_MANIFEST.json'))
    assert all(hashlib.sha256(code.read(n)).hexdigest()==h for n,h in hashes.items())
    out=dict(protocol='precommit-v2.2',seed=123,analysis_script_sha256=sha(Path(__file__)),code_zip_sha256=sha(args.runs/'code-v22.zip'),
             specification_sha256=v1.fingerprint(spec),dataset_sha256=v1.fingerprint(data),
             split_audit=v22.audit_splits(data),suite=suite,factorial={},patch={},audit={})
    all_patches={}
    for model,splits in [('illada',['calibration','discovery']),('dream',['discovery'])]:
        folder=args.runs/(model+'-v22');meta=load(folder/'manifest.json')
        assert meta['status']=='complete' and meta['seed']==123
        assert meta['specification_sha256']==v1.fingerprint(spec)
        assert meta['dataset_sha256']==v1.fingerprint(data)
        assert all(hashes[n]==h for n,h in meta['code_sha256'].items())
        for split in splits:
            stage=folder/split;gs=[g for g in data['groups'] if g['split']==split];lookup={g['id']:g for g in gs}
            freeze=load(stage/'frozen_before_inference.json')
            assert freeze['settings']==spec and freeze['specification_sha256']==v1.fingerprint(spec)
            assert freeze['target_ids']==list(lookup) and freeze['code_sha256']==meta['code_sha256']
            assert freeze['resample_sha256']==v1.fingerprint(v22.resample_pairs(gs))
            for name,digest in freeze['previous_files_sha256'].items():assert sha(folder/name)==digest
            rr=lines(stage/'factorial.jsonl');assert len(rr)==240
            assert len({(r['case'],r['requested'],r['slots'],r['unused']) for r in rr})==240
            for r in rr:
                assert r['p']==lookup[r['case']]['p']
                assert r['expected']=={k:v for k,v in v2.gold(r['p'],'downstream').items() if k in r['slots']}
                cc=r['canvas'];first=r['writes'][0]
                assert cc['prompt']==v2.prompt(r['p'],'downstream',r['requested'])
                assert [r['ids'][p] for p in cc['suffix_positions']]==cc['suffix_ids']
                assert set(first['eligible'])==set(p for ps in cc['spans'].values() for p in ps)
            fs={}
            for req in ('B','AB'):
                for slots in ('B','AB'):
                    for unused in ('pooled',False,True):
                        cell=[r for r in rr if r['requested']==req and r['slots']==slots and (unused=='pooled' or r['unused']==unused)]
                        fs[f'{req}/{slots}/{unused}']={stage:dict(n=len(cell),correct=sum(r[stage]['values']['B']==r['expected']['B'] if stage=='t0' else r['values']['B']==r['expected']['B'] for r in cell)) for stage in ('t0','final')}
            effects={}
            for timing in ('t0','final'):
                def correct(r):return (r['t0']['values'] if timing=='t0' else r['values'])['B']==r['expected']['B']
                for unused in ('pooled',False,True):
                    for req in ('B','AB'):
                        a=grouped([r for r in rr if r['requested']==req and r['slots']=='AB' and (unused=='pooled' or r['unused']==unused)],correct)
                        b=grouped([r for r in rr if r['requested']==req and r['slots']=='B' and (unused=='pooled' or r['unused']==unused)],correct)
                        effects[f'{timing}/slot_effect/{req}/{unused}']=contrast(a,b)
            out['factorial'][model+'/'+split]=dict(cells=fs,effects=effects,valid_t0=sum(r['t0']['valid'] for r in rr),valid_final=sum(r['valid'] for r in rr))
            if model!='illada':continue
            contexts=lines(stage/'contexts.jsonl');patches=lines(stage/'patch.jsonl')
            assert len(contexts)==660 and len(patches)==7440
            cx={c['context_id']:c for c in contexts};assert len(cx)==660
            pair_map={(p['Target'],p['repetition']):p['Source'] for p in v22.resample_pairs(gs)}
            source_readout={}
            for rel,ks in spec['relation_sources'].items():
                for kind in ks:
                    selected=[c for c in contexts if c['relation']==rel and c['source_kind']==kind and c['sign']!=0]
                    assert len(selected)==60 and len({(c['case'],c['sign']) for c in selected})==60
                    source_readout[rel+'/'+kind]=dict(n=60,**{k:sum(c['source_values'][k]==c['source_expected'][k] for c in selected) for k in ('A','B')})
            got=defaultdict(list);floor_hits=0;max_error=0;null_max=0
            for c in contexts:
                assert c['p']==lookup[c['case']]['p']
                if c['source_kind']!='resample':assert c['p_source']==v22.source(c['p'],c['source_kind'],c['sign'] or 1)
                else:
                    assert c['p_source']==lookup[c['source_case']]['p'] and c['source_case']!=c['case']
                    assert pair_map[c['case'],c['repetition']]==c['source_case']
                assert c['expected']==v2.gold(c['p'],c['relation'])
                assert c['source_expected']==v2.gold(c['p_source'],c['relation'])
                for cc in (c['Source'],c['Target']):
                    mask=set(cc['ids'][p] for ps in cc['spans'].values() for p in ps)
                    assert len(mask)==1 and all(cc['ids'][p] in mask for p in cc['unused'])
                    assert [cc['ids'][p] for p in cc['suffix_positions']]==cc['suffix_ids']
                assert c['Source']['spans']==c['Target']['spans']
                assert c['Source']['unused']==c['Target']['unused']
            def score(c,pred):
                nonlocal floor_hits
                by=dict(zip(c['prediction_positions'],pred));values=[];logp=[]
                for pos,a,z in zip(c['Target']['spans']['B'],c['B_target_token_ids'],c['B_candidate_token_ids']):
                    pa,pz=by[pos]['probabilities'][str(a)],by[pos]['probabilities'][str(z)]
                    floor_hits+=int(pa<1e-38)+int(pz<1e-38)
                    values.append(math.log(max(pz,1e-38))-math.log(max(pa,1e-38)))
                    logp.append(math.log(max(pa,1e-38)))
                return st.mean(values),st.mean(logp)
            for r in patches:
                c=cx[r['context_id']];got[r['context_id']].append({k:r[k] for k in ('mode','layers','site')})
                assert not set(r['target_positions'])&set(c['raw_query_positions'])
                for pred,values in [(c['baseline'],c['baseline_values']),(r['predictions'],r['values'])]:
                    by=dict(zip(c['prediction_positions'],pred))
                    ids=[by[p]['token_id'] for p in c['Target']['spans']['B']]
                    assert (ids==c['B_target_token_ids'])==(values['B']==c['expected']['B'])
                    assert (ids==c['B_candidate_token_ids'])==(values['B']==c['alternative']['B'])
                bs,blp=score(c,c['baseline']);ps,plp=score(c,r['predictions'])
                error=max(abs(bs-c['baseline_log_ratio']),abs(ps-r['log_ratio']),abs(ps-bs-r['delta']))
                max_error=max(max_error,error);assert error<1e-6
                if (r['source_kind']=='self' and r['site']!='A_to_unused') or r['layers']==[32]:
                    null_max=max(null_max,abs(r['delta']));assert abs(r['delta'])<=1e-6
                r.update(base_values=c['baseline_values'],expected=c['expected'],alternative=c['alternative'],
                         source_expected=c['source_expected'],target_logp_change=plp-blp)
            for cid,jobs in got.items():
                c=cx[cid]
                expected=([dict(mode='resample',layers=[31],site=s) for s in ('answer_A','unused')]
                          if c['sign']==0 else v22.schedule(c['relation'],c['source_kind']))
                assert jobs==expected,(cid,len(jobs),len(expected))
            assert set(got)==set(cx)
            grid={};group_means={}
            for r in patches:
                key='/'.join([r['relation'],r['source_kind'],r['mode'],str(r['layers'][0]),r['site']])
                grid.setdefault(key,[]).append(r)
            summaries={}
            for key,rs in grid.items():
                means=grouped(rs,lambda r:r['delta']);group_means[key]=means
                summaries[key]=dict(delta=estimate(means),n=len(rs),
                    accuracy_before=st.mean(r['base_values']['B']==r['expected']['B'] for r in rs),
                    accuracy_after=st.mean(r['values']['B']==r['expected']['B'] for r in rs),
                    accuracy_change=estimate(grouped(rs,lambda r:int(r['values']['B']==r['expected']['B'])-int(r['base_values']['B']==r['expected']['B']))),
                    candidate_before=sum(r['base_values']['B']==r['alternative']['B'] for r in rs),
                    candidate_after=sum(r['values']['B']==r['alternative']['B'] for r in rs),
                    target_logp_change=estimate(grouped(rs,lambda r:r['target_logp_change'])))
            def gm(rel,kind,mode='single',start=31,site='answer_A'):return group_means[f'{rel}/{kind}/{mode}/{start}/{site}']
            comp=contrast(gm('downstream','A_change_B_fixed'),gm('downstream','A_fixed_B_change'))
            d={k:gm('downstream','change_A')[k]-gm('downstream','same_A')[k] for k in lookup}
            i={k:gm('independent','change_A')[k]-gm('independent','same_A')[k] for k in lookup}
            contrasts=dict(compensation_minus_B_only=comp,downstream_change_minus_same=estimate(d),
                           independent_change_minus_same=estimate(i),dependency_difference=contrast(d,i))
            for start in (8,16,24):
                contrasts[f'sustained_{start}_minus_single31']=contrast(gm('downstream','change_A','sustained',start),gm('downstream','change_A'))
                contrasts[f'sustained_{start}_compensation_minus_B_only']=contrast(gm('downstream','A_change_B_fixed','sustained',start),gm('downstream','A_fixed_B_change','sustained',start))
            res=[r for r in patches if r['source_kind']=='resample' and r['site']=='answer_A']
            resample=dict(same_A_pairs=sum(r['source_expected']['A']==r['expected']['A'] for r in res),pairs=len(res),
                by_repetition={rep:dict(correct=sum(r['values']['B']==r['expected']['B'] for r in res if r['repetition']==rep),n=30) for rep in range(5)})
            out['patch'][split]=dict(grid=summaries,contrasts=contrasts,resample=resample,source_readout=source_readout)
            out['audit'][split]=dict(contexts=len(contexts),patches=len(patches),max_metric_error=max_error,floor_hits=floor_hits,null_max=null_max)
            all_patches[split]=patches
    # Calibration replay checks against completed v2.1, independent of the v2.2 implementation.
    old=ROOT/'runs/colab-precommit-v21/artifacts/runs/illada-v21'
    if old.exists():
        prior={(r['case'],r['requested'],r['slots'],r['unused']):r for r in lines(old/'factorial.jsonl')}
        new=lines(args.runs/'illada-v22/calibration/factorial.jsonl')
        for r in new:
            p=prior[r['case'],r['requested'],r['slots'],r['unused']]
            for k in ('t0','ids','values','writes'):assert r[k]==p[k],('Replay differs',r['case'],k)
        previous={(r['case'],r['source_kind'],r['sign'],r['layer'],r['site']):r for r in lines(old/'patch.jsonl')}
        count=0
        for r in all_patches['calibration']:
            key=(r['case'],r['source_kind'],r['sign'],r['layers'][0],r['site'])
            if r['relation']=='downstream' and r['mode']=='single' and key in previous:
                assert r['delta']==previous[key]['delta'] and r['predictions']==previous[key]['predictions'];count+=1
        out['audit']['v21_replay']=dict(factorial=240,patch=count)
    args.out.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(audit=out['audit'],contrasts={s:p['contrasts'] for s,p in out['patch'].items()}),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
