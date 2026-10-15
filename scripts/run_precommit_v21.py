"""Downstream 2x2 and iLLaDA A-to-B t=0 patching, seed 123."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dlm_coupling import precommit as v1, precommit_v2 as v2, precommit_v21 as v21
from run_precommit import MODELS, compact_canvas, dump


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=MODELS, default='illada')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-seconds', type=int, default=2400)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data = v2.dataset(args.seed)
    groups = [g for g in data['groups'] if g['split']=='calibration']
    dump(args.out/'dataset.json', {**data, 'groups': groups})
    cfg = dict(**MODELS[args.model], dtype='bfloat16', device=args.device, trust_remote_code=True)
    meta = dict(protocol='precommit-v2.1', status='preparing', seed=args.seed, model=cfg,
        dataset_sha256=v1.fingerprint(data),
        code_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted([Path(__file__), ROOT/'scripts/run_precommit.py', *ROOT.glob('src/**/*.py')])},
        analysis_status='Exploratory reuse of 30 v2 calibration groups; reserved groups untouched',
        design='downstream only; wording B/AB x slots B/AB x unused False/True; native ending; 240 outputs per model',
        patch_design='iLLaDA only; AB wording/slots; unused True; all 30 groups; signs -1/+1; change_A/same_A/self; four sites; fixed five-layer grid',
        decoder='v1 full-vocabulary confidence-greedy one answer token per forward; unused held masked; no carried hidden states',
        mismatch_policy='Fill every canvas slot; evaluate B. Off-diagonal wording/slot cells intentionally mismatch.',
        statistical_unit='underlying input group; average signs within group; 4000 group bootstrap draws seed 123',
        output_policy='t0 primary; unpatched rollout secondary; patched final rollout not measured',
        gate_policy='Exact patch baseline t0 A>=.8, B>=.5, format>=.95. No individual case filtering. Historical v2 gate unchanged.')
    def save():
        meta['elapsed_seconds'] = time.monotonic()-started
        dump(args.out/'manifest.json', meta)
    def emit(name, row):
        with (args.out/(name+'.jsonl')).open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    def budget():
        if time.monotonic()-started > args.max_seconds:
            raise TimeoutError('Configured process budget exceeded')
    save()
    if args.prepare_only:
        p = groups[1]['p']
        dump(args.out/'examples.json', dict(p=p, gold=v2.gold(p,'downstream'),
            prompts={r:v2.prompt(p,'downstream',r) for r in ('B','AB')},
            Sources={f'{kind}/{sign}':v21.source(p,kind,sign) for kind in v21.SOURCE_KINDS for sign in (-1,1)}))
        meta['status']='prepared_no_model_loaded';save();print('PREPARED', v1.fingerprint(data));return
    try:
        from dlm_coupling.backends.hf import HFBackend
        b = HFBackend(cfg, load_model=False)
        builder, old = v21.CanvasBuilder(b), v2.CanvasBuilder(b)
        jobs, sources = [], {}
        for g in groups:
            for requested in ('B','AB'):
                for slots in ('B','AB'):
                    for unused in (False,True):
                        cc = builder.build(g['p'], requested, slots, unused)
                        if requested==slots:
                            previous = old.build(g['p'],'downstream',requested,unused=unused)
                            for k in ('ids','spans','maps','unused','prompt','suffix_positions'):
                                assert previous[k]==cc[k], 'v2 baseline replay mismatch: '+k
                        jobs.append((g,requested,slots,unused,cc))
            if args.model=='illada':
                target = builder.build(g['p'])
                for sign in (-1,1):
                    for kind in v21.SOURCE_KINDS:
                        p_source = v21.source(g['p'],kind,sign)
                        src = builder.build(p_source)
                        v21.check_alignment(target,src)
                        ex, ex_source = v2.gold(g['p'],'downstream'), v2.gold(p_source,'downstream')
                        assert ex_source=={k:v+(sign if kind=='change_A' else 0) for k,v in ex.items()}
                        assert p_source['v']==g['p']['v'] and p_source['s']==g['p']['s']
                        sources[g['id'],kind,sign] = (p_source,src)
        meta.update(status='tokenizer_audit_passed',audit=dict(canvases=len(jobs),
            source_alignments=len(sources),domain=[10,99],
            native_suffixes=sorted({cc['native_suffix'] for *_,cc in jobs})), backend=b.metadata)
        save();print('TOKENIZER_AUDIT',json.dumps(meta['audit']),flush=True)
        b._load_model(); b._torch.manual_seed(args.seed)
        counts = dict(forwards=0,processed_token_positions=0)
        original_predict = b.predict
        def counted(ids,positions,probes):
            budget()
            result = original_predict(ids,positions,probes)
            counts['forwards']+=1;counts['processed_token_positions']+=len(ids)
            return result
        b.predict=counted
        meta.update(status='running',backend=b.metadata,compute=counts);save()
        baseline_rows=[]
        for g,requested,slots,unused,cc in jobs:
            expected={k:v for k,v in v2.gold(g['p'],'downstream').items() if k in slots}
            rr=v1.decode(b,cc)
            first=rr['writes'][0]
            t0=v21.prediction_values(b,cc,first['eligible'],first['predictions'])
            assert all(rr['ids'][p]==b.mask_id for p in cc['unused'])
            assert [rr['ids'][p] for p in cc['suffix_positions']]==cc['suffix_ids']
            row=dict(kind='baseline',case=g['id'],p=g['p'],relation='downstream',requested=requested,
                slots=slots,unused=unused,expected=expected,t0=t0,canvas=compact_canvas(cc),**rr)
            emit('factorial',row);baseline_rows.append(row)
            if requested==slots=='AB' and unused:print('FACTORIAL_DONE',g['id'],flush=True)
        gate=v21.baseline_gate(baseline_rows,len(groups))
        layers=v1.layer_grid(len(b._model.model.layers))
        meta.update(gate=gate,layers_1based=[x+1 for x in layers],outputs=dict(factorial=len(baseline_rows),patch=0))
        save();print('PATCH_BASELINE',json.dumps(gate),flush=True)
        if args.model=='illada' and gate['pass_gate']:
            for g in groups:
                target=builder.build(g['p']);expected=v2.gold(g['p'],'downstream')
                for sign in (-1,1):
                    for kind in v21.SOURCE_KINDS:
                        p_source,src=sources[g['id'],kind,sign]
                        context_id=f"{g['id']}/{kind}/{sign}"
                        emit('contexts',dict(id=context_id,p=g['p'],p_source=p_source,
                            Target=compact_canvas(target),Source=compact_canvas(src),
                            expected=expected,source_expected=v2.gold(p_source,'downstream')))
                        for patch in v21.scan(b,target,src,expected,sign,layers,kind):
                            emit('patch',dict(context_id=context_id,case=g['id'],sign=sign,
                                 source_kind=kind,expected=expected,**patch))
                            meta['outputs']['patch']+=1
                save();print('PATCH_DONE',g['id'],meta['outputs']['patch'],flush=True)
        meta['status']='complete' if args.model!='illada' or gate['pass_gate'] else 'baseline_below_feasibility'
        save();print('COMPLETE',args.model,json.dumps(dict(status=meta['status'],outputs=meta['outputs'],compute=counts)),flush=True)
    except Exception as exc:
        meta.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
