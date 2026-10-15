"""Run the paired ending diagnostic and the prespecified addition-only calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dlm_coupling import precommit as v1, precommit_v2 as v2
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
    dump(args.out/'reserved_hashes.json', [dict(split=g['split'],sha256=g['instance_sha256'])
                                         for g in data['groups'] if g['split']!='calibration'])
    cfg = dict(**MODELS[args.model], dtype='bfloat16', device=args.device, trust_remote_code=True)
    meta = dict(protocol='precommit-v2', status='preparing', seed=args.seed, model=cfg,
        dataset_sha256=v1.fingerprint(data),
        code_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted([Path(__file__), ROOT/'scripts/run_precommit.py', *ROOT.glob('src/**/*.py')])},
        decoder='same as v1: full vocabulary; greedy highest-confidence single answer token each forward; no carried hidden states',
        tail_diagnostic='iLLaDA only: same 10 v1 groups; numeric; A/B; both unused variants; bare/native endings, 240 outputs',
        calibration='30 new groups; additive; numeric; native ending fixed before outcomes; AB/BA/A/B; both unused variants, 720 outputs',
        gate='single requests pooled across unused conditions, point >= .8, operational pooled Wilson lower >= .7, format >= .95; six cells',
        statistical_unit='underlying input group; all relations/requests/canvases paired',
        intervention_policy='no interventions in this calibration script')
    def save():
        meta['elapsed_seconds'] = time.monotonic()-started
        dump(args.out/'manifest.json', meta)
    def budget():
        if time.monotonic()-started > args.max_seconds:
            raise TimeoutError('Configured process budget exceeded')
    save()
    if args.prepare_only:
        dump(args.out/'examples.json',[dict(relation=rel, prompt=v2.prompt(groups[0]['p'],rel),
            gold=v2.gold(groups[0]['p'],rel),source=v2.source(groups[0]['p'],rel,'diff_C_same_answers'))
            for rel in v2.RELATIONS])
        meta['status']='prepared_no_model_loaded';save();print('PREPARED',v1.fingerprint(data));return
    try:
        from dlm_coupling.backends.hf import HFBackend
        b = HFBackend(cfg, load_model=False)
        b2 = v2.CanvasBuilder(b)
        jobs = []
        if args.model=='illada':
            old = [g for g in v1.dataset(args.seed)['groups'] if g['split']=='calibration']
            legacy = v2.CanvasBuilder(b, task='legacy')
            original = v1.CanvasBuilder(b)
            for g in old:
                for rel in v2.RELATIONS:
                    for req in ('A','B'):
                        for unused in (False,True):
                            for ending in ('bare','native'):
                                cc = legacy.build(g['p'],rel,req,'numeric',unused,ending)
                                if ending=='bare':
                                    cc1=original.build(g['p'],rel,req,'numeric',unused)
                                    for key in ('ids','spans','maps','unused','prompt'):
                                        assert cc[key]==cc1[key],f'v1 replay canvas mismatch: {key}'
                                jobs.append(('ending',g,rel,req,unused,ending,cc,v1.gold(g['p'],rel)))
        for g in groups:
            for rel in v2.RELATIONS:
                for req in ('AB','BA','A','B'):
                    for unused in (False,True):
                        cc = b2.build(g['p'],rel,req,'numeric',unused,'native')
                        jobs.append(('calibration',g,rel,req,unused,'native',cc,v2.gold(g['p'],rel)))
        # All real canvases audit 10..99; audit all five Source alignments on
        # three prespecified groups, including joint/single and unused variants.
        alignments = 0
        for g in groups[:3]:
            for rel in v2.RELATIONS:
                for req in ('AB','BA','A','B'):
                    for unused in (False,True):
                        cc = b2.build(g['p'],rel,req,'numeric',unused)
                        for kind in v2.SOURCE_KINDS:
                            sc = b2.build(v2.source(g['p'],rel,kind),rel,req,'numeric',unused)
                            v1.check_alignment(cc,sc)
                            assert cc['suffix_positions']==sc['suffix_positions']
                            alignments += 1
        meta.update(status='tokenizer_audit_passed', audit=dict(canvases=len(jobs),
            alignments=alignments,alignment_groups=3,domain=[10,99],
            native_suffixes=sorted({cc['native_suffix'] for *_,cc,_ in jobs if cc['ending']=='native'})),backend=b.metadata)
        save(); print('TOKENIZER_AUDIT',json.dumps(meta['audit']),flush=True)
        b._load_model();b._torch.manual_seed(args.seed)
        counts=dict(forwards=0,processed_token_positions=0)
        original_predict=b.predict
        def counted(ids, positions, probes):
            budget()
            rr=original_predict(ids, positions, probes)
            counts['forwards']+=1;counts['processed_token_positions']+=len(ids)
            return rr
        b.predict=counted
        meta.update(status='running',backend=b.metadata,compute=counts);save()
        calibration=[]
        for phase,g,rel,req,unused,ending,cc,expected in jobs:
            budget()
            rr=v1.decode(b,cc)
            expected={k:expected[k] for k in req}
            assert all(rr['ids'][p]==b.mask_id for p in cc['unused'])
            assert [rr['ids'][p] for p in cc['suffix_positions']]==cc['suffix_ids']
            row=dict(kind='baseline',phase=phase,case=g['id'],p=g['p'],relation=rel,
                requested=req,style='numeric',unused=unused,ending=ending,source_kind='self',
                expected=expected,correct=rr['values']==expected,canvas=compact_canvas(cc),**rr)
            with (args.out/f'{phase}.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if phase=='calibration':calibration.append(row)
            if rel=='downstream' and req=='B' and unused and ending=='native':
                print('FINISHED',phase,g['id'],flush=True)
        gate=v2.score_gate(calibration,expected_groups=len(groups))
        meta.update(status='calibrated_v2' if gate['pass_gate'] else 'calibration_below_gate_v2',
                    gate=gate,outputs=dict(calibration=len(calibration),ending=len(jobs)-len(calibration)))
        save();print('COMPLETE',args.model,meta['status'],json.dumps(gate),flush=True)
    except Exception as exc:
        meta.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':
    main()
