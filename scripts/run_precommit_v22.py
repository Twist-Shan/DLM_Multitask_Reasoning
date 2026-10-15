"""v2.2 robustness on calibration30, then frozen replication on discovery30."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dlm_coupling import precommit as v1, precommit_v2 as v2, precommit_v21 as v21, precommit_v22 as v22
from run_precommit import MODELS, compact_canvas, dump


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=MODELS, default='illada')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--max-seconds', type=int, default=3600)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data, spec = v2.dataset(args.seed), v22.specification(args.seed)
    audit = v22.audit_splits(data)
    splits = ['calibration','discovery'] if args.model=='illada' else ['discovery']
    groups_by_split = {s:[g for g in data['groups'] if g['split']==s] for s in splits}
    dump(args.out/'dataset.json', {**data,'groups':[g for s in splits for g in groups_by_split[s]]})
    dump(args.out/'specification.json', spec)
    pairs = {s:v22.resample_pairs(gs,args.seed) for s,gs in groups_by_split.items()}
    dump(args.out/'resample_pairs.json', pairs)
    cfg = dict(**MODELS[args.model], dtype='bfloat16', device=args.device, trust_remote_code=True)
    meta = dict(protocol='precommit-v2.2',status='preparing',seed=args.seed,model=cfg,
        dataset_sha256=v1.fingerprint(data), specification_sha256=v1.fingerprint(spec), split_audit=audit,
        code_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted([Path(__file__),ROOT/'scripts/run_precommit.py',*ROOT.glob('src/**/*.py')])},
        outputs={}, compute=dict(forwards=0,processed_token_positions=0),
        scope='All prespecified controls; no accuracy-based case exclusion; no confirmation100 inference')
    def save():
        meta['elapsed_seconds'] = time.monotonic()-started
        dump(args.out/'manifest.json',meta)
    def emit(split,name,row):
        with (args.out/split/(name+'.jsonl')).open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        meta['outputs'][split][name] = meta['outputs'][split].get(name,0)+1
    save()
    if args.prepare_only:
        meta['status']='prepared_no_model_loaded';save()
        print('PREPARED',json.dumps(dict(audit=audit,specification_sha256=meta['specification_sha256'])),flush=True)
        return
    try:
        from dlm_coupling.backends.hf import HFBackend
        b = HFBackend(cfg,load_model=False)
        builder,factorial = v22.CanvasBuilder(b),v21.CanvasBuilder(b)
        # Audit all tokenizer alignments without looking at model outcomes.
        for gs in groups_by_split.values():
            for g in gs:
                for req in ('B','AB'):
                    for slots in ('B','AB'):
                        for unused in (False,True):factorial.build(g['p'],req,slots,unused)
                if args.model=='illada':
                    for rel,ks in spec['relation_sources'].items():
                        cc = builder.build(g['p'],rel)
                        if rel=='downstream':
                            old = factorial.build(g['p'])
                            for key in ('ids','maps','spans','unused','prompt_A','suffix_positions'):
                                assert old[key]==cc[key],key
                        for kind in ks:
                            for sign in (-1,1):
                                v21.check_alignment(cc,builder.build(v22.source(g['p'],kind,sign),rel))
        print('TOKENIZER_AUDIT_PASSED',args.model,flush=True)
        b._load_model();b._torch.manual_seed(args.seed)
        assert len(b._model.model.layers)==32 if args.model=='illada' else True
        original = b.predict
        def counted(ids,positions,probes):
            if time.monotonic()-started>args.max_seconds:raise TimeoutError('Process time budget')
            result=original(ids,positions,probes)
            meta['compute']['forwards']+=1;meta['compute']['processed_token_positions']+=len(ids)
            return result
        b.predict=counted
        meta.update(status='running',backend=b.metadata);save()
        for split,gs in groups_by_split.items():
            (args.out/split).mkdir()
            meta['outputs'][split]={}
            # Freeze all settings before the first discovery forward; no adaptive selection.
            freeze = dict(specification_sha256=meta['specification_sha256'],code_sha256=meta['code_sha256'],
                stage=split,settings=spec,resample_sha256=v1.fingerprint(pairs[split]),
                target_ids=[g['id'] for g in gs],past_forwards=meta['compute']['forwards'],
                previous_files_sha256={p.relative_to(args.out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted((args.out/'calibration').glob('*.jsonl'))} if split=='discovery' else {})
            dump(args.out/split/'frozen_before_inference.json',freeze)
            print('BEGIN_SPLIT',split,'FROZEN',v1.fingerprint(freeze),flush=True)
            base_rows=[]
            for g in gs:
                for req in ('B','AB'):
                    for slots in ('B','AB'):
                        for unused in (False,True):
                            cc=factorial.build(g['p'],req,slots,unused)
                            rr=v1.decode(b,cc);first=rr['writes'][0]
                            t0=v21.prediction_values(b,cc,first['eligible'],first['predictions'])
                            expected={k:v for k,v in v2.gold(g['p'],'downstream').items() if k in slots}
                            assert all(rr['ids'][p]==b.mask_id for p in cc['unused'])
                            row=dict(case=g['id'],p=g['p'],relation='downstream',requested=req,slots=slots,
                                     unused=unused,expected=expected,t0=t0,canvas=compact_canvas(cc),**rr)
                            emit(split,'factorial',row);base_rows.append(row)
            meta.setdefault('baseline_diagnostics',{})[split]=v21.baseline_gate(base_rows,len(gs))
            print('BASELINE',split,json.dumps(meta['baseline_diagnostics'][split]),flush=True);save()
            if args.model=='illada':
                def run_context(g,q,rel,kind,sign,jobs,extra=None):
                    target,src=builder.build(g['p'],rel),builder.build(q,rel)
                    expected=v2.gold(g['p'],rel)
                    alternative={k:v+sign for k,v in expected.items()}
                    if kind=='resample':
                        alternative=dict(A=v2.gold(q,rel)['A'],B=v2.gold(q,rel)['A']+g['p']['v']+g['p']['s'])
                    context,rows=v22.evaluate(b,target,src,expected,alternative,jobs,kind)
                    cid=f"{g['id']}/{rel}/{kind}/{sign}/{(extra or {}).get('repetition','none')}"
                    identity=dict(context_id=cid,case=g['id'],relation=rel,source_kind=kind,sign=sign,**(extra or {}))
                    emit(split,'contexts',dict(**identity,p=g['p'],p_source=q,Target=compact_canvas(target),
                         Source=compact_canvas(src),source_expected=v2.gold(q,rel),**context))
                    for row in rows:emit(split,'patch',dict(**identity,**row))
                for g in gs:
                    for rel,ks in spec['relation_sources'].items():
                        for kind in ks:
                            for sign in (-1,1):
                                run_context(g,v22.source(g['p'],kind,sign),rel,kind,sign,v22.schedule(rel,kind))
                    save();print('PATCH_DONE',split,g['id'],meta['outputs'][split]['patch'],flush=True)
                lookup={g['id']:g for g in gs}
                jobs=[dict(mode='resample',layers=[31],site=s) for s in ('answer_A','unused')]
                for pair in pairs[split]:
                    run_context(lookup[pair['Target']],lookup[pair['Source']]['p'],'downstream','resample',0,jobs,
                                dict(repetition=pair['repetition'],source_case=pair['Source']))
                for g in gs:
                    run_context(g,g['p'],'downstream','self',0,jobs,dict(repetition='self',source_case=g['id']))
                assert meta['outputs'][split]==dict(factorial=240,contexts=660,patch=7440),meta['outputs'][split]
            print('SPLIT_COMPLETE',split,json.dumps(meta['outputs'][split]),flush=True);save()
        meta['status']='complete';save()
        print('COMPLETE',args.model,json.dumps(dict(outputs=meta['outputs'],compute=meta['compute'])),flush=True)
    except Exception as exc:
        meta.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
