"""Original-task replay and controlled order, space, and redundant-definition tests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling import precommit as v1,precommit_v2 as v2,precommit_v25 as v
from run_precommit import MODELS,compact_canvas,dump


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--audit-only',action='store_true')
    ap.add_argument('--limit',type=int,default=30)
    ap.add_argument('--max-seconds',type=int,default=5400)
    args=ap.parse_args()
    if not 1<=args.limit<=30:raise ValueError('limit must be 1..30')
    args.out.mkdir(parents=True,exist_ok=False)
    data,spec=v.dataset(args.seed),v.specification(args.seed)
    groups=data['groups'][:args.limit]
    dump(args.out/'dataset.json',dict(data,groups=groups));dump(args.out/'specification.json',spec)
    cfg=dict(**MODELS['illada'],dtype='bfloat16',device='cuda',trust_remote_code=True)
    meta=dict(protocol=v.PROTOCOL,status='preparing',seed=args.seed,groups=len(groups),model=cfg,
        dataset_sha256=v1.fingerprint(data),specification_sha256=v1.fingerprint(spec),outputs={},
        code_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__),ROOT/'scripts/run_precommit.py',*sorted(ROOT.glob('src/**/*.py'))]},
        compute=dict(forwards=0,processed_token_positions=0,prediction_seconds=0.))
    start=time.monotonic()
    def save():
        meta['elapsed_seconds']=time.monotonic()-start;dump(args.out/'manifest.json',meta)
    def emit(name,row):
        with (args.out/(name+'.jsonl')).open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        meta['outputs'][name]=meta['outputs'].get(name,0)+1
    save()
    if args.prepare_only:
        p=groups[0]['p'];item=v.redundant(p)
        dump(args.out/'examples.json',dict(original=[dict(relation=rel,prompt=v2.prompt(p,rel,'AB','numeric'),
             expected=v2.gold(p,rel)) for rel in v2.RELATIONS],redundant=[dict(parent=parent,
             prompt=v.redundant_prompt(item,parent),expected=v.redundant_gold(item,parent)) for parent in ('A','R')]))
        meta['status']='prepared_no_model_loaded';save();print('PREPARED',v1.fingerprint(data));return
    try:
        from dlm_coupling.backends.hf import HFBackend
        b=HFBackend(cfg,load_model=False);builder=v.CanvasBuilder(b);old=v2.CanvasBuilder(b)
        audits=anchors=0
        for g in groups:
            p=g['p']
            for relation,layout,quantity in spec['core_conditions']:
                cc=builder.core(p,relation,layout,quantity)
                base=builder.core(p,relation,layout,'mask_2')
                if cc['prompt']!=base['prompt'] or cc['spans']!=base['spans']:
                    raise ValueError('Space sweep changed question/answer positions')
                if layout=='AB' and quantity=='mask_2':
                    prior=old.build(p,relation,'AB','numeric',True,'native')
                    for key in ('ids','spans','maps','unused','suffix_positions','native_suffix'):
                        if cc[key]!=prior[key]:raise ValueError('Original anchor mismatch: '+key)
                    anchors+=1
            for relation in ('downstream','independent'):
                for layout in ('AB','BA'):
                    cc=builder.core(p,relation,layout)
                    for kind in v.core_kinds(relation):
                        for sign in ((1,) if kind=='self' else (-1,1)):
                            v.check_alignment(cc,builder.core(v.core_source(p,kind,sign),relation,layout));audits+=1
            item=v.redundant(p)
            for parent in ('A','R'):
                left,right=(builder.alias(item,parent,layout) for layout in ('ABR','RBA'))
                if left['spans']['B']!=right['spans']['B'] or len(left['ids'])!=len(right['ids']):
                    raise ValueError('Redundant spatial control moved B')
                builder.alias(item,parent,'AB')
                for layout in ('ABR','RBA'):
                    cc=builder.alias(item,parent,layout)
                    for kind in v.ALIAS_KINDS:
                        for sign in ((1,) if kind=='self' else (-1,1)):
                            ss=builder.alias(v.redundant_source(item,parent,kind,sign),parent,layout)
                            v.check_alignment(cc,ss);audits+=1
        meta.update(status='tokenizer_audit_passed',tokenizer_alignments=audits,exact_old_anchors=anchors,backend=b.metadata)
        save();print('TOKENIZER_AUDIT_PASSED',audits,'EXACT_OLD_ANCHORS',anchors,flush=True)
        if args.audit_only:return
        b._load_model();b._torch.manual_seed(args.seed)
        if len(b._model.model.layers)!=32:raise ValueError('Expected pinned 32-layer iLLaDA')
        predict=b.predict
        def counted(ids,positions,probes):
            if time.monotonic()-start>args.max_seconds:raise TimeoutError('Run budget exceeded')
            t=time.monotonic();result=predict(ids,positions,probes)
            meta['compute']['forwards']+=1;meta['compute']['processed_token_positions']+=len(ids)
            meta['compute']['prediction_seconds']+=time.monotonic()-t
            return result
        b.predict=counted;meta.update(status='running',backend=b.metadata);save()
        dump(args.out/'frozen_before_inference.json',dict(spec=spec,manifest=meta))
        def behavior(name,identity,cc,expected):
            positions=[pos for ps in cc['spans'].values() for pos in ps]
            pred=b.predict(cc['ids'],positions,[[] for _ in positions])
            t0=v.read_prediction(b,cc,positions,pred)
            for policy in spec['policies']:
                t=time.monotonic();before=meta['compute']['forwards']
                result=v.decode(b,cc,policy)
                emit(name,dict(**identity,expected=expected,canvas=compact_canvas(cc),t0=t0,
                     forwards=meta['compute']['forwards']-before,seconds=time.monotonic()-t,**result))
        # Finish every behavior condition first. This records ability before patch effects.
        for g in groups:
            p=g['p'];item=v.redundant(p)
            for relation,layout,quantity in spec['core_conditions']:
                ident=dict(case=g['id'],relation=relation,layout=layout,quantity=quantity)
                behavior('core_behavior',ident,builder.core(p,relation,layout,quantity),v2.gold(p,relation))
            for parent in ('A','R'):
                for layout in spec['alias_layouts']:
                    ident=dict(case=g['id'],parent=parent,layout=layout)
                    expected={k:value for k,value in v.redundant_gold(item,parent).items() if k in layout}
                    behavior('alias_behavior',ident,builder.alias(item,parent,layout),expected)
            save();print('BEHAVIOR_DONE',g['id'],json.dumps(meta['outputs']),flush=True)
        for g in groups:
            p=g['p'];item=v.redundant(p)
            for relation in ('downstream','independent'):
                for layout in ('AB','BA'):
                    cc=builder.core(p,relation,layout);expected=v2.gold(p,relation)
                    for kind in v.core_kinds(relation):
                        for sign in ((1,) if kind=='self' else (-1,1)):
                            q=v.core_source(p,kind,sign);ss=builder.core(q,relation,layout)
                            context,rows=v.patch_scan(b,cc,ss,expected,sign,v.core_schedule(kind),kind,
                                                     regions=dict(A=cc['spans']['A'],unused=cc['unused']))
                            ident=dict(case=g['id'],relation=relation,layout=layout,kind=kind,sign=sign,
                                       context_id=f"core/{g['id']}/{relation}/{layout}/{kind}/{sign}")
                            emit('core_contexts',dict(**ident,p=p,source_p=q,source_expected=v2.gold(q,relation),
                                 Target=compact_canvas(cc),Source=compact_canvas(ss),**context))
                            for row in rows:emit('core_patch',dict(**ident,**row))
                    if relation=='downstream':
                        for sign in (-1,1):
                            ss=builder.core(v.core_source(p,'A_change_B_fixed',sign),relation,layout)
                            for mode in ('first','persistent'):
                                rr=v.decode(b,cc,'B_first',ss,mode,cc['spans']['A'],v.MODES['L24_31'])
                                emit('core_rollout',dict(case=g['id'],relation=relation,layout=layout,sign=sign,
                                    kind='A_change_B_fixed',expected=expected,candidate_B=expected['B']+sign,
                                    canvas=compact_canvas(cc),**rr))
            for parent in ('A','R'):
                expected=v.redundant_gold(item,parent)
                for layout in ('ABR','RBA'):
                    cc=builder.alias(item,parent,layout)
                    for kind in v.ALIAS_KINDS:
                        for sign in ((1,) if kind=='self' else (-1,1)):
                            q=v.redundant_source(item,parent,kind,sign);ss=builder.alias(q,parent,layout)
                            context,rows=v.patch_scan(b,cc,ss,expected,sign,v.alias_schedule(kind),kind,
                                regions=dict(A=cc['spans']['A'],R=cc['spans']['R'],unused=cc['unused']))
                            ident=dict(case=g['id'],parent=parent,layout=layout,kind=kind,sign=sign,
                                parent_side='left' if layout.index(parent)<layout.index('B') else 'right',
                                context_id=f"alias/{g['id']}/{parent}/{layout}/{kind}/{sign}")
                            emit('alias_contexts',dict(**ident,p=item,source_p=q,source_expected=v.redundant_gold(q,parent),
                                 Target=compact_canvas(cc),Source=compact_canvas(ss),**context))
                            for row in rows:emit('alias_patch',dict(**ident,**row))
            save();print('PATCH_DONE',g['id'],json.dumps(meta['outputs']),flush=True)
        planned={key:n*len(groups) for key,n in spec['planned_per_group'].items()}
        if meta['outputs']!=planned:raise AssertionError((meta['outputs'],planned))
        meta['status']='complete';save()
        print('COMPLETE',json.dumps(dict(outputs=meta['outputs'],compute=meta['compute'])),flush=True)
    except Exception as exc:
        meta.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
