"""Validate and summarize v2.1; all inferential units are the 30 base groups."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling import precommit as v1, precommit_v2 as v2, precommit_v21 as v21


def read(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def estimate(values):
    values=list(values);rng=random.Random(123)
    draws=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(4000))
    return dict(mean=statistics.mean(values),ci95=[draws[99],draws[3899]],n_groups=len(values))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--v2',type=Path,default=ROOT/'runs/colab-precommit-v2/artifacts/runs')
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    ds=v2.dataset(123);groups=[g for g in ds['groups'] if g['split']=='calibration']
    cases=sorted(g['id'] for g in groups);by_case={g['id']:g['p'] for g in groups}
    result=dict(protocol='precommit-v2.1',seed=123,exploratory=True,models={})
    for model in ('illada','dream'):
        run=args.runs/(model+'-v21')
        if model=='dream' and (args.runs/'dream-v21-retry').exists():run=args.runs/'dream-v21-retry'
        meta=json.loads((run/'manifest.json').read_text())
        assert meta['status']=='complete' and meta['seed']==123
        assert meta['dataset_sha256']==v1.fingerprint(ds)
        archived_code={}
        original=args.runs/'code-v21-original.zip'
        if original.exists():
            assert hashlib.sha256(original.read_bytes()).hexdigest()=='39ea84f408a8a7b3169ef85794125b1744d447621da63f4a22197d75866a571d'
            with zipfile.ZipFile(original) as z:archived_code={name:z.read(name) for name in z.namelist()}
            if model=='dream':
                fix=args.runs/'code-v21-tokenizer-fix.zip'
                assert hashlib.sha256(fix.read_bytes()).hexdigest()=='5dbbc0fbdffa0bdce186885fab784d1b09f89c4103bb6ad95367fac160f5d7c5'
                with zipfile.ZipFile(fix) as z:archived_code.update({name:z.read(name) for name in z.namelist()})
        for name,digest in meta['code_sha256'].items():
            actual=archived_code[name] if archived_code else (ROOT/name).read_bytes()
            assert hashlib.sha256(actual).hexdigest()==digest,name
        rows=read(run/'factorial.jsonl');assert len(rows)==240
        index={(r['case'],r['requested'],r['slots'],r['unused']):r for r in rows}
        assert len(index)==240
        old={(r['case'],r['requested'],r['unused']):r for r in read(args.v2/(model+'-v2/calibration.jsonl')) if r['relation']=='downstream'}
        replay=0
        for case in cases:
            for request in ('B','AB'):
                for slots in ('B','AB'):
                    for unused in (False,True):
                        r=index[case,request,slots,unused];cc=r['canvas'];p=by_case[case]
                        assert r['p']==p and r['expected']=={k:v2.gold(p,'downstream')[k] for k in slots}
                        assert cc['prompt']==v2.prompt(p,'downstream',request)
                        assert set(cc['spans'])==set(slots) and not cc['uses_gold_to_prepare']
                        mask_id=meta['backend']['mask_token_id'] if 'mask_token_id' in meta['backend'] else (5 if model=='illada' else 151666)
                        assert all(cc['ids'][j]==mask_id for ps in cc['spans'].values() for j in ps)
                        assert all(r['ids'][j]==mask_id for j in cc['unused'])
                        assert [r['ids'][j] for j in cc['suffix_positions']]==cc['suffix_ids']
                        first=r['writes'][0];assert first['state']==cc['ids']
                        preds=dict(zip(first['eligible'],first['predictions']))
                        for field,ps in cc['spans'].items():
                            text=''.join(preds[j]['token_text'] for j in ps)
                            assert text==r['t0']['texts'][field]
                        if request==slots:
                            previous=old[case,request,unused]
                            assert previous['canvas']['ids']==cc['ids']
                            assert previous['ids']==r['ids']
                            assert previous['writes'][0]['predictions']==first['predictions']
                            replay+=1
        assert v21.baseline_gate(rows)==meta['gate']
        cells={}
        for request in ('B','AB'):
            for slots in ('B','AB'):
                for unused in (False,True,'pooled'):
                    rr=[r for r in rows if r['requested']==request and r['slots']==slots and (unused=='pooled' or r['unused']==unused)]
                    cells[f'{request}/{slots}/{unused}']=dict(n=len(rr),
                        t0_B=sum(r['t0']['values']['B']==r['expected']['B'] for r in rr),
                        final_B=sum(r['values']['B']==r['expected']['B'] for r in rr),
                        t0_A=sum(r['t0']['values'].get('A')==r['expected'].get('A') for r in rr) if slots=='AB' else None)
        contrasts={}
        for stage in ('t0','final'):
            def correct(case,request,slots,unused):
                r=index[case,request,slots,unused]
                return float((r['t0'] if stage=='t0' else r)['values']['B']==r['expected']['B'])
            def value(case,request,slots):return sum(correct(case,request,slots,u) for u in (False,True))/2
            for request in ('B','AB'):
                contrasts[f'{stage}/slot_effect_at_request_{request}']=estimate(value(c,request,'AB')-value(c,request,'B') for c in cases)
            for slots in ('B','AB'):
                contrasts[f'{stage}/wording_effect_at_slots_{slots}']=estimate(value(c,'AB',slots)-value(c,'B',slots) for c in cases)
            contrasts[f'{stage}/interaction']=estimate(value(c,'AB','AB')-value(c,'AB','B')-value(c,'B','AB')+value(c,'B','B') for c in cases)
        result['models'][model]=dict(cells=cells,contrasts=contrasts,gate=meta['gate'],identical_v2_replays=replay,
            compute=meta['compute'],elapsed_seconds=meta['elapsed_seconds'])
    run=args.runs/'illada-v21';patches=read(run/'patch.jsonl');assert len(patches)==3600
    contexts={r['id']:r for r in read(run/'contexts.jsonl')};assert len(contexts)==180
    pi={(r['case'],r['sign'],r['source_kind'],r['layer'],r['site']):r for r in patches}
    assert len(pi)==3600
    layers=[8,16,24,31,32];floor_terms=0
    for case in cases:
        for sign in (-1,1):
            for kind in v21.SOURCE_KINDS:
                for layer in layers:
                    for site in v21.SITES:
                        r=pi[case,sign,kind,layer,site];ctx=contexts[r['context_id']]
                        target,src=ctx['Target'],ctx['Source']
                        v21.check_alignment(target,src)
                        assert ctx['p_source']==v21.source(ctx['p'],kind,sign)
                        assert ctx['source_expected']==v2.gold(ctx['p_source'],'downstream')
                        assert r['alternative']=={k:v+sign for k,v in r['expected'].items()}
                        regions=dict(answer_A=target['spans']['A'],unused=target['unused'],
                            A_to_unused=target['unused'],prompt_A=target['prompt_A'])
                        assert r['target_positions']==regions[site]
                        assert not set(r['target_positions'])&set(r['raw_query_positions'])
                        assert all(target['ids'][j]==5 for ps in target['spans'].values() for j in ps)
                        assert all(src['ids'][j]==5 for ps in src['spans'].values() for j in ps)
                        if layer==32 or (kind=='self' and site!='A_to_unused'):assert abs(r['delta'])<=1e-6
                        def score(predictions):
                            nonlocal floor_terms
                            pp=dict(zip(r['prediction_positions'],predictions));terms=[]
                            for pos,a,b in zip(target['spans']['B'],r['B_target_token_ids'],r['B_cf_token_ids']):
                                pa,pb=pp[pos]['probabilities'][str(a)],pp[pos]['probabilities'][str(b)]
                                floor_terms+=int(pa<1e-38)+int(pb<1e-38)
                                terms.append(math.log(max(pb,1e-38))-math.log(max(pa,1e-38)))
                            return statistics.mean(terms)
                        base,after=score(r['baseline']),score(r['predictions'])
                        assert abs(base-r['baseline_log_ratio'])<1e-10
                        assert abs(after-base-r['delta'])<1e-10
    grid={};paired={}
    for layer in layers:
        for site in v21.SITES:
            for kind in v21.SOURCE_KINDS:
                rr=[r for r in patches if r['layer']==layer and r['site']==site and r['source_kind']==kind]
                e=estimate(statistics.mean(pi[c,s,kind,layer,site]['delta'] for s in (-1,1)) for c in cases)
                e.update(n_interventions=len(rr),positive=sum(r['delta']>1e-6 for r in rr),
                    cf_after=sum(r['B_after']==r['alternative']['B'] for r in rr),
                    target_after=sum(r['B_after']==r['expected']['B'] for r in rr),
                    mean_patch_l2=statistics.mean(r['patch_l2'] for r in rr))
                def base_ids(r):
                    pred=dict(zip(r['prediction_positions'],r['baseline']))
                    return [pred[j]['token_id'] for j in contexts[r['context_id']]['Target']['spans']['B']]
                e['cf_before']=sum(base_ids(r)==r['B_cf_token_ids'] for r in rr)
                e['target_before']=sum(base_ids(r)==r['B_target_token_ids'] for r in rr)
                e['new_cf_hits']=sum(base_ids(r)!=r['B_cf_token_ids'] and r['B_after']==r['alternative']['B'] for r in rr)
                e['lost_cf_hits']=sum(base_ids(r)==r['B_cf_token_ids'] and r['B_after']!=r['alternative']['B'] for r in rr)
                grid[f'{layer}/{site}/{kind}']=e
            for control in ('same_A','self'):
                paired[f'{layer}/{site}/change_A-minus-{control}']=estimate(statistics.mean(
                    pi[c,s,'change_A',layer,site]['delta']-pi[c,s,control,layer,site]['delta'] for s in (-1,1)) for c in cases)
        for control in ('unused','A_to_unused','prompt_A'):
            paired[f'{layer}/change_A/answer_A-minus-{control}']=estimate(statistics.mean(
                pi[c,s,'change_A',layer,'answer_A']['delta']-pi[c,s,'change_A',layer,control]['delta'] for s in (-1,1)) for c in cases)
    result['patch']=dict(records=len(patches),contexts=len(contexts),grid=grid,paired=paired,
        probability_floor_uses=floor_terms,
        null_max_abs_delta=max(abs(r['delta']) for r in patches if r['layer']==32 or (r['source_kind']=='self' and r['site']!='A_to_unused')),
        interpretation='One forward only; all 30 groups; signs averaged within group; pointwise exploratory intervals; no multiple-comparison correction')
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(models={m:dict(cells=v['cells'],contrasts=v['contrasts']) for m,v in result['models'].items()},patch_grid=grid,patch_null=result['patch']['null_max_abs_delta']),indent=2))


if __name__=='__main__':main()
