"""Run the frozen ARB/RAB and directed-attention exploration on calibration30."""
import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling import precommit as v1,precommit_v2 as v2,precommit_v25 as old,precommit_v26 as v
from run_precommit import MODELS,compact_canvas,dump


def prediction_difference(a,b):
    return max(abs(x['probabilities'][key]-y['probabilities'][key])
               for x,y in zip(a,b) for key in x['probabilities'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--limit',type=int,default=30);ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--max-seconds',type=int,default=5400);args=ap.parse_args()
    if not 1<=args.limit<=30:raise ValueError('limit')
    args.out.mkdir(parents=True,exist_ok=False)
    data,spec=v.dataset(args.seed),v.specification(args.seed);groups=data['groups'][:args.limit]
    dump(args.out/'dataset.json',dict(data,groups=groups));dump(args.out/'specification.json',spec)
    cfg=dict(**MODELS['illada'],dtype='bfloat16',device='cuda',trust_remote_code=True)
    meta=dict(protocol=v.PROTOCOL,status='preparing',seed=args.seed,groups=len(groups),model=cfg,
        dataset_sha256=v1.fingerprint(data),specification_sha256=v1.fingerprint(spec),outputs={},
        code_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__),ROOT/'scripts/run_precommit.py',*sorted(ROOT.glob('src/**/*.py'))]},
        compute=dict(forwards=0,processed_token_positions=0,prediction_seconds=0.))
    started=time.monotonic()
    def save():
        meta['elapsed_seconds']=time.monotonic()-started;dump(args.out/'manifest.json',meta)
    def emit(name,row):
        with (args.out/(name+'.jsonl')).open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        meta['outputs'][name]=meta['outputs'].get(name,0)+1
    save()
    if args.prepare_only:
        item=old.redundant(groups[0]['p'])
        dump(args.out/'examples.json',[dict(parent=p,prompt=old.redundant_prompt(item,p),gold=old.redundant_gold(item,p)) for p in spec['parents']])
        meta['status']='prepared_no_model_loaded';save();print('PREPARED',spec['planned_per_group']);return
    try:
        from dlm_coupling.backends.hf import HFBackend
        b=HFBackend(cfg);builder=v.CanvasBuilder(b);prior=old.CanvasBuilder(b);audits=anchors=0
        for g in groups:
            item=old.redundant(g['p'])
            for layout in v.LAYOUTS:
                for parent in spec['parents']:
                    cc=builder.alias(item,parent,layout)
                    for kind in v.KINDS:
                        for sign in ((1,) if kind=='self' else (-1,1)):
                            old.check_alignment(cc,builder.alias(old.redundant_source(item,parent,kind,sign),parent,layout));audits+=1
                aa,rr=(builder.alias(item,p,layout) for p in spec['parents'])
                if aa['spans']!=rr['spans'] or len(aa['ids'])!=len(rr['ids']) or sum(x!=y for x,y in zip(aa['ids'],rr['ids']))!=1:
                    raise ValueError('Parent switch changed geometry or multiple tokens')
            for layout,quantity,mode in spec['attention_conditions']:
                cc=builder.core(g['p'],'downstream',layout,quantity)
                ref=prior.core(g['p'],'downstream',layout,quantity)
                if cc!=ref:raise ValueError('v2.5 input changed')
                v.attention_plan(cc,mode);anchors+=1
        meta.update(tokenizer_alignments=audits,exact_v25_anchors=anchors);save()
        b._load_model();b._torch.manual_seed(args.seed);layers=b._model.model.layers
        if len(layers)!=32:raise ValueError('Expected 32 layers')
        meta['attention_classes']=sorted({type(layer.self_attn).__name__ for layer in layers})
        if meta['attention_classes']!=['ILLaDASdpaAttention']:raise ValueError('Keep the official SDPA implementation')
        meta['attention_source_sha256']=hashlib.sha256(Path(inspect.getsourcefile(type(layers[0].self_attn))).read_bytes()).hexdigest()
        base_predict=b.predict
        def counted(ids,positions,probes):
            if time.monotonic()-started>args.max_seconds:raise TimeoutError('GPU run budget')
            t=time.monotonic();p=base_predict(ids,positions,probes)
            meta['compute']['forwards']+=1;meta['compute']['processed_token_positions']+=len(ids)
            meta['compute']['prediction_seconds']+=time.monotonic()-t;return p
        b.predict=counted;meta.update(status='running',backend=b.metadata);save()
        dump(args.out/'frozen_before_inference.json',dict(specification=spec,manifest=meta))
        # Model-level directional test, before substantive data collection.
        pp=groups[0]['p'];expected=v2.gold(pp,'downstream')
        cm,ce=(builder.core(pp,'downstream','AB',q) for q in ('mask_32','eos_32'))
        pos,probes=v1.probes_for(cm,expected,dict(expected,B=expected['B']+1))
        native=b.predict(cm['ids'],pos,probes)
        with v.AttentionEdges(layers,cm,'sham') as h:sham=b.predict(cm['ids'],pos,probes)
        h.audit(1)
        blocked=[]
        for cc in (cm,ce):
            with v.AttentionEdges(layers,cc,'tail_to_original') as h:blocked.append(b.predict(cc['ids'],pos,probes))
            h.audit(1)
        restored=b.predict(cm['ids'],pos,probes)
        preflight=dict(native_sham_probe_max_difference=prediction_difference(native,sham),
            blocked_mask_eos_probe_max_difference=prediction_difference(*blocked),
            blocked_mask_eos_top_equal=[p['token_id'] for p in blocked[0]]==[p['token_id'] for p in blocked[1]],
            restored_native_probe_max_difference=prediction_difference(native,restored))
        dump(args.out/'attention_preflight.json',preflight)
        if preflight['blocked_mask_eos_probe_max_difference']>1e-6 or not preflight['blocked_mask_eos_top_equal'] or preflight['restored_native_probe_max_difference']>1e-6:
            raise AssertionError(('Directional attention preflight failed',preflight))
        print('PREFLIGHT',json.dumps(preflight),flush=True)
        def behavior(name,identity,cc,expected,mode=None):
            pos,probes=v1.probes_for(cc,expected,dict(expected,B=expected['B']+1))
            hook=v.AttentionEdges(layers,cc,mode) if mode else None
            with hook if hook else nullcontext():
                pred=b.predict(cc['ids'],pos,probes);t0=old.read_prediction(b,cc,pos,pred);total=1
                for policy in v.POLICIES:
                    result=old.decode(b,cc,policy);total+=result['nfe']
                    record=dict(**identity,expected=expected,canvas=compact_canvas(cc),t0=t0,
                        positions=pos,t0_predictions=pred,gold_token_ids={k:cc['maps'][k][value] for k,value in expected.items()},
                        forwards=result['nfe'],**result)
                    if hook:record.update(attention=hook.audit(total),hook_forward_count=total)
                    emit(name,record)
        for g in groups:
            p=g['p'];item=old.redundant(p)
            for parent in spec['parents']:
                for layout in v.LAYOUTS:
                    behavior('alias_behavior',dict(case=g['id'],parent=parent,layout=layout),
                             builder.alias(item,parent,layout),old.redundant_gold(item,parent))
            for layout,quantity,mode in spec['attention_conditions']:
                behavior('attention_behavior',dict(case=g['id'],relation='downstream',layout=layout,quantity=quantity,attention_mode=mode),
                         builder.core(p,'downstream',layout,quantity),v2.gold(p,'downstream'),mode)
            save();print('BEHAVIOR_DONE',g['id'],json.dumps(meta['outputs']),flush=True)
        for g in groups:
            item=old.redundant(g['p'])
            for parent in spec['parents']:
                expected=old.redundant_gold(item,parent)
                for layout in v.LAYOUTS:
                    cc=builder.alias(item,parent,layout)
                    for kind in v.KINDS:
                        for sign in ((1,) if kind=='self' else (-1,1)):
                            q=old.redundant_source(item,parent,kind,sign);ss=builder.alias(q,parent,layout)
                            context,rows=old.patch_scan(b,cc,ss,expected,sign,v.schedule(kind),kind,
                                regions=dict(A=cc['spans']['A'],R=cc['spans']['R'],unused=cc['unused']))
                            ident=dict(case=g['id'],parent=parent,layout=layout,kind=kind,sign=sign,
                                       context_id=f"alias/{g['id']}/{parent}/{layout}/{kind}/{sign}")
                            emit('alias_contexts',dict(**ident,p=item,source_p=q,source_expected=old.redundant_gold(q,parent),
                                Target=compact_canvas(cc),Source=compact_canvas(ss),**context))
                            for row in rows:emit('alias_patch',dict(**ident,**row))
            save();print('PATCH_DONE',g['id'],json.dumps(meta['outputs']),flush=True)
        planned={k:n*len(groups) for k,n in spec['planned_per_group'].items()}
        if meta['outputs']!=planned:raise AssertionError((meta['outputs'],planned))
        meta.update(status='complete',backend=b.metadata);save()
        print('COMPLETE',json.dumps(dict(outputs=meta['outputs'],compute=meta['compute'])),flush=True)
    except Exception as exc:
        meta.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
