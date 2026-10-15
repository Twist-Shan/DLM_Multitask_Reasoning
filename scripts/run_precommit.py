"""Prepare, audit, calibrate, then run pre-commitment discovery (never confirmation)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling.precommit import (CanvasBuilder, RELATIONS, SOURCE_KINDS, dataset, gold,
    source, counterfactual, fingerprint, decode, scan, layer_grid, score_gate)
from dlm_coupling.shared_causal import ResidualPatch

MODELS={
    'illada':dict(family='illada',model_id='GSAI-ML/iLLaDA-8B-Instruct',
                  revision='5769f049224b2cff61106ebce0429ede5deeb434'),
    'dream':dict(family='dream',model_id='Dream-org/Dream-v0-Instruct-7B',
                 revision='05334cb9faaf763692dcf9d8737c642be2b2a6ae')}
STYLES=('symbolic','numeric')


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def compact_canvas(cc):
    return {k:v for k,v in cc.items() if k!='maps'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('prepare','audit','calibrate','discover'),required=True)
    ap.add_argument('--model',choices=MODELS,default='illada')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--calibration',type=Path)
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--max-seconds',type=int,default=2700)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    data=dataset(args.seed)
    cfg=dict(**MODELS[args.model],dtype='bfloat16',device=args.device,trust_remote_code=True)
    meta=dict(status='preparing',phase=args.phase,seed=args.seed,model=cfg,
        protocol='precommit-v1',dataset_sha256=fingerprint(data),streams=data['streams'],
        split_sizes=dict(calibration=10,discovery=30,confirmation=100),
        code_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted([Path(__file__),*ROOT.glob('src/**/*.py')])},
        decoder='full-vocabulary greedy; one most-confident answer token per forward; leftmost tie; no carried hidden state',
        canvas='fixed labels, all requested answer positions eligible; extra unused tail masks held masked when present',
        discovery_event='t=0; all answer positions masked in Source and Target',
        probability_metric='mean per-token same-state log ratio, not joint sequence likelihood',
        candidate_signature='A+1, B+2 for arithmetic in every relation; B+2 is an off-graph diagnostic in independent tasks',
        statistical_unit='underlying eight-input group; layers, relations and Sources are paired repeated measurements',
        scope='preparation or diagnostic discovery; no confirmatory inference; no GPU outcomes bundled')
    started=time.monotonic()
    def save():
        meta['elapsed_seconds']=time.monotonic()-started
        dump(args.out/'manifest.json',meta)
    def emit(row):
        with (args.out/'results.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    def budget():
        if time.monotonic()-started>args.max_seconds: raise TimeoutError('Configured process budget exceeded')
    save()
    # Confirmation hashes are reserved, but its prompts are not inspected or run.
    dump(args.out/'dataset.json',{**data,'groups':[g for g in data['groups'] if g['split']!='confirmation']})
    dump(args.out/'reserved_confirmation_hashes.json',[g['instance_sha256'] for g in data['groups'] if g['split']=='confirmation'])
    if args.phase=='prepare':
        from dlm_coupling.precommit import prompt
        sample=data['groups'][0]['p']
        dump(args.out/'examples.json',[dict(relation=rel,requested=req,
            prompt=prompt(sample,rel,req),gold=gold(sample,rel))
            for rel in RELATIONS for req in ('AB','BA','A','B')])
        meta['status']='prepared_no_model_loaded';save();print(json.dumps(meta,indent=2));return
    try:
        calibration=None
        if args.phase=='discover':
            if not args.calibration: raise ValueError('--calibration is required')
            calibration=json.loads((args.calibration/'manifest.json').read_text(encoding='utf-8'))
            if calibration['status']!='calibrated' or not calibration['gate']['selected_style']:
                raise ValueError('Calibration gate has not passed; do not run discovery')
            if calibration['dataset_sha256']!=meta['dataset_sha256'] or calibration['model']!=cfg:
                raise ValueError('Calibration dataset/model mismatch')
            if calibration['code_sha256']!=meta['code_sha256']:
                raise ValueError('Code changed after calibration; recalibrate under a recorded protocol revision')
            meta['calibration_manifest_sha256']=hashlib.sha256((args.calibration/'manifest.json').read_bytes()).hexdigest()
        from dlm_coupling.backends.hf import HFBackend
        b=HFBackend(cfg,load_model=False)
        builder=CanvasBuilder(b)
        split='discovery' if args.phase=='discover' else 'calibration'
        groups=[g for g in data['groups'] if g['split']==split]
        styles=(calibration['gate']['selected_style'],) if calibration else STYLES
        # Tokenizer-only phase never loads weights. Invalid layouts are explicit failures.
        if args.phase=='audit':
            counts=dict(canvases=0,alignments=0)
            from dlm_coupling.precommit import check_alignment
            for group in groups:
                for rel in RELATIONS:
                    for style in styles:
                        for req in ('AB','BA','A','B'):
                            for unused in (False,True):
                                cc=builder.build(group['p'],rel,req,style,unused);counts['canvases']+=1
                                for kind in SOURCE_KINDS:
                                    ss=builder.build(source(group['p'],rel,kind),rel,req,style,unused)
                                    check_alignment(cc,ss);counts['alignments']+=1
            meta.update(status='tokenizer_audit_passed',audit=counts,backend=b.metadata)
            save();print(json.dumps(counts));return
        b._load_model();b._torch.manual_seed(args.seed)
        counts=dict(forwards=0,processed_token_positions=0)
        original_predict=b.predict
        def counted_predict(ids,positions,probes):
            budget()
            result=original_predict(ids,positions,probes)
            counts['forwards']+=1;counts['processed_token_positions']+=len(ids)
            return result
        b.predict=counted_predict
        meta['compute']=counts
        meta.update(status='running',backend=b.metadata);save()
        cache={}
        def baseline(group,rel,req,style,unused,kind='self'):
            budget()
            key=(group['id'],rel,req,style,unused,kind)
            if key not in cache:
                p=source(group['p'],rel,kind)
                cc=builder.build(p,rel,req,style,unused)
                rr=decode(b,cc)
                expected={k:gold(p,rel)[k] for k in req}
                record=dict(kind='baseline',case=group['id'],relation=rel,requested=req,
                    style=style,unused=unused,source_kind=kind,expected=expected,
                    correct=rr['values']==expected,canvas=compact_canvas(cc),**rr)
                emit(record);cache[key]=(cc,record)
            return cache[key]
        if args.phase=='calibrate':
            records=[]
            for style in styles:
                for g in groups:
                    for rel in RELATIONS:
                        for req in ('AB','BA','A','B'):
                            for unused in (False,True):
                                records.append(baseline(g,rel,req,style,unused)[1])
                    print('CALIBRATED',style,g['id'],flush=True)
            gate=score_gate(records,styles)
            meta.update(gate=gate,status='calibrated' if gate['selected_style'] else 'calibration_below_gate')
            save();print('GATE',gate['selected_style'],flush=True);return
        style=styles[0];layers=layer_grid(len(b._model.model.layers))
        # Rollout layer is fixed a priori at the middle layer, not the scan winner.
        rollout_layer=round(len(b._model.model.layers)*.5)-1
        meta.update(style=style,layers_one_based=[l+1 for l in layers],
                    rollout_layer_one_based=rollout_layer+1);save()
        for group in groups:
            for rel in RELATIONS:
                for req in ('AB','BA'):
                    target,tr=baseline(group,rel,req,style,True)
                    expected=gold(group['p'],rel)
                    alt=dict(A=expected['A']+1,B=expected['B']+2)
                    for kind in SOURCE_KINDS:
                        src,sr=baseline(group,rel,req,style,True,kind)
                        for direction in ('BA','AB'):
                            budget()
                            # Self gives an implementation check in addition to controls.
                            rr=scan(b,target,src,direction,expected,alt,layers)
                            for r in rr:
                                if kind=='self' and r['site']=='answer_'+direction[0]:
                                    assert r['predictions']==r['baseline'],'Self patch changes predictions'
                                if r['layer']==len(b._model.model.layers):
                                    pos=[p for ps in target['spans'].values() for p in ps]
                                    query_indices=[pos.index(p) for p in target['spans'][direction[1]]]
                                    assert all(r['predictions'][j]==r['baseline'][j] for j in query_indices), 'Final-layer spatial null failed'
                                emit(dict(kind='scan',case=group['id'],relation=rel,requested=req,
                                    source_kind=kind,eligible=tr['correct'] and sr['correct'],
                                    target_gold=expected,source_gold=gold(source(group['p'],rel,kind),rel),
                                    expected_cf=counterfactual(group['p'],rel,kind),**r))
                            # Paired native/fixed order rollouts from t=0; one-forward patch.
                            if kind in ('self','diff_C_same_answers'):
                                origin=direction[0]
                                module=b._model.model.layers[rollout_layer]
                                with ResidualPatch(module,src['spans'][origin]) as cap:
                                    b.predict(src['ids'],target['spans'][direction[1]],[[] for _ in target['spans'][direction[1]]])
                                fixed=[w['position'] for w in tr['writes']]
                                for schedule,order in [('native',None),('fixed_baseline_positions',fixed)]:
                                    control=decode(b,target,fixed_order=order)
                                    assert control['ids']==tr['ids'],'Baseline schedule replay failed'
                                    for site_name,ps in [('answer_'+origin,target['spans'][origin]),('unused',target['unused'])]:
                                        result=decode(b,target,fixed_order=order,
                                            patch_context=lambda:ResidualPatch(module,ps,cap.captured))
                                        if kind=='self' and site_name=='answer_'+origin:
                                            assert result['ids']==control['ids'],'Self rollout changed'
                                        emit(dict(kind='rollout',case=group['id'],relation=rel,requested=req,
                                            direction=direction,source_kind=kind,site=site_name,schedule=schedule,
                                            layer=rollout_layer+1,patch_window='first_forward_only',
                                            baseline_values=control['values'],baseline_order=fixed,
                                            expected_cf=counterfactual(group['p'],rel,kind),**result))
                # True single-objective and equal-extra-mask behavioral controls.
                for req in ('A','B'):
                    for unused in (False,True):baseline(group,rel,req,style,unused)
            print('DISCOVERY',group['id'],flush=True);save()
        # A pure visible-number copying control, kept out of arithmetic aggregation.
        for i,q in enumerate(range(20,30)):
            g=dict(id=f'copy-{i:02d}',p=dict(q=q))
            cc,tr=baseline(g,'copy','AB',style,True)
            ss,sr=baseline(g,'copy','AB',style,True,'C_only')
            for direction in ('BA','AB'):
                for r in scan(b,cc,ss,direction,gold(g['p'],'copy'),dict(A=q+1,B=q+1),layers):
                    emit(dict(kind='copy_scan',case=g['id'],relation='copy',requested='AB',
                        source_kind='C_only',eligible=tr['correct'] and sr['correct'],**r))
        meta['status']='discovery_complete';save()
    except Exception as exc:
        meta.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
