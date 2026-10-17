"""Bounded iLLaDA exploration: dependency, order, capacity; no confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling import precommit as v1, precommit_v24 as v24
from run_precommit import MODELS,compact_canvas,dump


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--audit-only',action='store_true')
    ap.add_argument('--max-seconds',type=int,default=5400)
    ap.add_argument('--limit',type=int,default=30,help='Smoke testing only; main run is all 30')
    args=ap.parse_args()
    if not 1<=args.limit<=30:raise ValueError('--limit must be 1..30')
    args.out.mkdir(parents=True,exist_ok=False)
    data,spec=v24.dataset(args.seed),v24.specification(args.seed)
    groups=data['groups'][:args.limit]
    dump(args.out/'dataset.json',{**data,'groups':groups});dump(args.out/'specification.json',spec)
    cfg=dict(**MODELS['illada'],dtype='bfloat16',device='cuda',trust_remote_code=True)
    meta=dict(protocol=v24.PROTOCOL,status='preparing',seed=args.seed,model=cfg,
              groups=len(groups),dataset_sha256=v1.fingerprint(data),specification_sha256=v1.fingerprint(spec),
              code_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__),ROOT/'scripts/run_precommit.py',*sorted(ROOT.glob('src/**/*.py'))]},
              compute=dict(forwards=0,processed_token_positions=0,prediction_seconds=0.),outputs={})
    start=time.monotonic()
    def save():
        meta['elapsed_seconds']=time.monotonic()-start;dump(args.out/'manifest.json',meta)
    def emit(name,row):
        with (args.out/(name+'.jsonl')).open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        meta['outputs'][name]=meta['outputs'].get(name,0)+1
    save()
    if args.prepare_only:
        p=groups[0]['p']
        dump(args.out/'examples.json',[dict(parent=k,prompt=v24.prompt(p,k),gold=v24.gold(p,k)) for k in ('A','C')])
        meta['status']='prepared_no_model_loaded';save();print('PREPARED',json.dumps(meta['outputs']));return
    try:
        from dlm_coupling.backends.hf import HFBackend
        b=HFBackend(cfg,load_model=False);builder=v24.CanvasBuilder(b)
        audits=0
        for g in groups:
            for parent in spec['parents']:
                left=builder.build(g['p'],parent,'ABC');right=builder.build(g['p'],parent,'CBA')
                if len(left['ids'])!=len(right['ids']) or left['spans']['B']!=right['spans']['B']:
                    raise ValueError('B moved in spatial comparison')
                for layout in spec['layouts']:
                    cc=builder.build(g['p'],parent,layout)
                    for kind in spec['kinds']:
                        for sign in (-1,1):
                            ss=builder.build(v24.source(g['p'],parent,kind,sign),parent,layout)
                            v24.check_alignment(cc,ss);audits+=1
                base=builder.space(g['p'],parent,'mask_0')
                for cond in spec['space_quantity']+spec['space_usage']:
                    cc=builder.space(g['p'],parent,cond)
                    if cc['spans']['B']!=base['spans']['B'] or cc['prompt']!=base['prompt']:
                        raise ValueError('B position or wording changed in space sweep')
                    for kind in ('comp_'+parent,'same_'+parent,'self'):
                        for sign in (-1,1):
                            ss=builder.space(v24.source(g['p'],parent,kind,sign),parent,cond)
                            v24.check_alignment(cc,ss);audits+=1
        meta.update(status='tokenizer_audit_passed',tokenizer_alignments=audits,backend=b.metadata);save()
        print('TOKENIZER_AUDIT_PASSED',audits,flush=True)
        if args.audit_only:return
        b._load_model();b._torch.manual_seed(args.seed)
        if len(b._model.model.layers)!=32:raise ValueError('Expected pinned 32-layer iLLaDA')
        original=b.predict
        def counted(ids,positions,probes):
            if time.monotonic()-start>args.max_seconds:raise TimeoutError('Bounded run budget exceeded')
            before=time.monotonic();result=original(ids,positions,probes)
            meta['compute']['forwards']+=1;meta['compute']['processed_token_positions']+=len(ids)
            meta['compute']['prediction_seconds']+=time.monotonic()-before
            return result
        b.predict=counted;meta.update(status='running',backend=b.metadata);save()
        dump(args.out/'frozen_before_inference.json',dict(spec=spec,manifest=meta))
        for g in groups:
            p=g['p']
            for parent in spec['parents']:
                expected=v24.gold(p,parent)
                for layout in spec['layouts']:
                    cc=builder.build(p,parent,layout)
                    ident=dict(case=g['id'],equal_AC=g['equal_AC'],parent=parent,layout=layout,
                               parent_side='left' if layout.index(parent)<layout.index('B') else 'right')
                    positions=[q for ps in cc['spans'].values() for q in ps]
                    pred=b.predict(cc['ids'],positions,[[] for _ in positions])
                    t0=v24.read_prediction(b,cc,positions,pred)
                    for policy in ('confidence','B_first'):
                        before=meta['compute']['forwards'];wall=time.monotonic()
                        rr=v24.decode(b,cc,policy)
                        emit('behavior',dict(**ident,expected=expected,canvas=compact_canvas(cc),t0=t0,
                                             forwards=meta['compute']['forwards']-before,
                                             seconds=time.monotonic()-wall,**rr))
                    for kind in spec['kinds']:
                        signs=(1,) if kind=='self' else (-1,1)
                        for sign in signs:
                            q=v24.source(p,parent,kind,sign);ss=builder.build(q,parent,layout)
                            context,rows=v24.patch_scan(b,cc,ss,expected,sign,v24.schedule(kind),kind)
                            cid=f"{g['id']}/{parent}/{layout}/{kind}/{sign}"
                            details=dict(**ident,context_id=cid,kind=kind,sign=sign)
                            emit('contexts',dict(**details,p=p,source_p=q,source_expected=v24.gold(q,parent),
                                                 Target=compact_canvas(cc),Source=compact_canvas(ss),**context))
                            for row in rows:emit('patch',dict(**details,**row))
                    # Same B-first baseline for both single-forward and persistent interventions.
                    for sign in (-1,1):
                        kind='comp_'+parent
                        ss=builder.build(v24.source(p,parent,kind,sign),parent,layout)
                        for mode in ('first','persistent'):
                            rr=v24.decode(b,cc,'B_first',ss,mode,cc['spans'][parent],v24.MODES['L24_31'])
                            emit('rollout',dict(**ident,kind=kind,sign=sign,expected=expected,
                                 candidate_B=expected['B']+sign,canvas=compact_canvas(cc),**rr))
                for cond in spec['space_quantity']+spec['space_usage']:
                    cc=builder.space(p,parent,cond)
                    positions=cc['spans']['B'];pred=b.predict(cc['ids'],positions,[[] for _ in positions])
                    t0=v24.read_prediction(b,cc,positions,pred)
                    before=meta['compute']['forwards'];wall=time.monotonic()
                    rr=v24.decode(b,cc,'B_first',only_B=True)
                    ident=dict(case=g['id'],equal_AC=g['equal_AC'],parent=parent,condition=cond)
                    emit('space',dict(**ident,expected=expected,canvas=compact_canvas(cc),t0=t0,
                          forwards=meta['compute']['forwards']-before,seconds=time.monotonic()-wall,**rr))
                    if cond in spec['space_usage']:
                        for kind in ('comp_'+parent,'same_'+parent,'self'):
                            for sign in ((1,) if kind=='self' else (-1,1)):
                                q=v24.source(p,parent,kind,sign);ss=builder.space(q,parent,cond)
                                jobs=[dict(site='tail',mode=m,layers=v24.MODES[m]) for m in ('L31','L24_31')]
                                context,rows=v24.patch_scan(b,cc,ss,expected,sign,jobs,kind,
                                                           regions=dict(tail=cc['patch_positions']))
                                cid=f"{g['id']}/{parent}/{cond}/{kind}/{sign}"
                                emit('space_contexts',dict(**ident,context_id=cid,kind=kind,sign=sign,
                                     p=p,source_p=q,source_expected=v24.gold(q,parent),
                                     Target=compact_canvas(cc),Source=compact_canvas(ss),**context))
                                for row in rows:emit('space_patch',dict(**ident,context_id=cid,kind=kind,sign=sign,**row))
            save();print('GROUP_DONE',g['id'],json.dumps(meta['outputs']),flush=True)
        n=len(groups)
        planned=dict(behavior=8*n,contexts=60*n,patch=196*n,rollout=16*n,
                     space=22*n,space_contexts=40*n,space_patch=80*n)
        if meta['outputs']!=planned:raise AssertionError((meta['outputs'],planned))
        meta['status']='complete';meta['backend']=b.metadata;save()
        print('COMPLETE',json.dumps(dict(outputs=meta['outputs'],compute=meta['compute'])),flush=True)
    except Exception as exc:
        meta.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
