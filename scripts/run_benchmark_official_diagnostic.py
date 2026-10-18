"""Four exploratory controls using a pinned, separately obtained author decoder.

This is not an official benchmark evaluation: prompts and the 1024-token cap
are from our v2 adaptation. Both commit rate and termination guidance change.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling import benchmark_transfer as base, benchmark_transfer_v2 as task
from run_precommit import MODELS

UPSTREAM_REVISION='9182493720ed723ef8031210d85959364e51cbe0'
UPSTREAM_SHA256='d4d2d3c015511e63fe9237cd3197346148fb0c795b6c89e0b5ef4846be25252e'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--upstream',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--prompt-version',choices=('v2','v3'),default='v2')
    args=ap.parse_args()
    if args.prompt_version=='v3':
        from dlm_coupling import benchmark_transfer_v3 as prompt_task
    else:
        prompt_task=task
    if hashlib.sha256(args.upstream.read_bytes()).hexdigest()!=UPSTREAM_SHA256:
        raise ValueError('Author decoder hash mismatch; obtain generate.py at pinned revision')
    data=json.loads(args.dataset.read_text(encoding='utf-8'))
    cases=task.selected_cases(data,1)
    jobs=[dict(case_id=c['id'],benchmark=c['benchmark'],requested=r,prompt=prompt_task.prompt(c,r))
          for c in cases if c['hops']==2 for r in ('B','AB')]
    args.out.mkdir(parents=True,exist_ok=False)
    def dump(name,value):
        (args.out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=dict(protocol='benchmark-author-decoder-diagnostic',status='prepared',seed=123,
        prompt_version=args.prompt_version,prompt_protocol=prompt_task.PROTOCOL,
        prompt_module_sha256=hashlib.sha256(Path(prompt_task.__file__).read_bytes()).hexdigest(),
        upstream_revision=UPSTREAM_REVISION,upstream_sha256=UPSTREAM_SHA256,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset_sha256=base.digest(data),gen_length=1024,block_length=32,steps_per_block=32,
        end_think_logit_boost=100,end_think_boost_power=3,rows=0,
        interpretation='Exploratory post-v2 decoder diagnostic; not official benchmark scores')
    dump('manifest.json',manifest);dump('jobs.json',jobs);dump('dataset.json',data)
    spec=importlib.util.spec_from_file_location('author_generate',args.upstream)
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    from dlm_coupling.backends.hf import HFBackend
    cfg=dict(**MODELS['illada'],dtype='bfloat16',device='cuda',trust_remote_code=True)
    backend=HFBackend(cfg);backend._load_model()
    torch,model,tok=backend._torch,backend._model,backend.tokenizer
    torch.manual_seed(123)
    count=dict(forwards=0,processed_token_positions=0)
    def hook(module,args):
        count['forwards']+=1;count['processed_token_positions']+=int(args[0].numel())
    handle=model.register_forward_pre_hook(hook)
    end_ids=tok.encode('</think>',add_special_tokens=False)
    eos=tok.eos_token_id
    eos_ids=set(eos if isinstance(eos,list) else [eos])-{None}
    stops=[tok.decode([i],skip_special_tokens=False) for i in eos_ids]
    manifest.update(status='running',model=cfg,end_think_token_ids=end_ids,stop_tokens=stops)
    dump('manifest.json',manifest)
    by_id={c['id']:c for c in cases}
    started=time.monotonic()
    try:
        for job in jobs:
            text=tok.apply_chat_template([dict(role='user',content=job['prompt'])],tokenize=False,add_generation_prompt=True)
            ids=tok.encode(text,add_special_tokens=False)
            if backend.mask_id in ids or len(ids)+1024>8192:
                raise ValueError('Invalid prompt/context budget')
            prompt=torch.tensor([ids],device=model.device)
            before=dict(count);tick=time.monotonic()
            out=upstream.var_generate(model,tok,prompt,steps=32,gen_length=1024,block_length=32,
                temperature=0.,cfg_scale=0.,remasking='low_confidence',mask_id=backend.mask_id,
                stop_tokens=stops,end_think_token_ids=end_ids,end_think_logit_boost=100.,end_think_boost_power=3.)
            assert out[0,:len(ids)].tolist()==ids
            output=out[0,len(ids):].tolist()
            end=next((i for i,t in enumerate(output) if t in eos_ids),len(output))
            generation=dict(text=tok.decode(output[:end],skip_special_tokens=False,clean_up_tokenization_spaces=False),
                raw_text=tok.decode(output,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                output_token_ids=output,effective_output_tokens=end,eos_found=end<len(output),
                mask_remaining=sum(t==backend.mask_id for t in output),gold_used_for_generation=False)
            row=dict(**job,generation=generation,compute={k:count[k]-before[k] for k in count},
                seconds=time.monotonic()-tick,**task.score(by_id[job['case_id']],generation['text'],job['requested']))
            with (args.out/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            manifest.update(rows=manifest['rows']+1,elapsed_seconds=time.monotonic()-started)
            dump('manifest.json',manifest)
            print(json.dumps({k:row[k] for k in ('benchmark','requested','format_valid','scores','seconds')}),flush=True)
        manifest['status']='complete'
    except Exception as exc:
        manifest.update(status='failed_partial',error=repr(exc));raise
    finally:
        handle.remove();manifest.update(compute=count,elapsed_seconds=time.monotonic()-started)
        dump('manifest.json',manifest)


if __name__=='__main__':main()
