"""Small qualitative free-generation screen; no supplied intermediate values."""
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling.implicit_sharing import data, prompt, gold, RELATIONS
from dlm_coupling.backends.hf import HFBackend
from dlm_coupling.text_generation import generate_text, generate_blockwise

def main():
    out=ROOT/'runs/free-reasoning';out.mkdir(parents=True,exist_ok=False)
    cases=[c for c in data(123) if c['split']=='validation'][:2]
    cfg=dict(family='illada',model_id='GSAI-ML/iLLaDA-8B-Instruct',revision='5769f049224b2cff61106ebce0429ede5deeb434',trust_remote_code=True,dtype='bfloat16',device='cuda')
    b=HFBackend(cfg,load_model=True);b._torch.manual_seed(123)
    manifest=dict(status='running',seed=123,cases=cases,model=cfg,backend=b.metadata,
        selection='first two previously generated validation input groups, without filtering for correctness; qualitative follow-up, not fresh held-out evaluation',
        max_new_tokens=512,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    started=time.monotonic();nfe=0;count=0
    def save():
        manifest.update(nfe=nfe,rows=count,elapsed_seconds=time.monotonic()-started)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save()
    with (out/'results.jsonl').open('w',encoding='utf-8') as f:
        try:
            for style in ('numeric','symbolic'):
                for c in cases:
                    for rel in RELATIONS if style=='numeric' else ('shared',):
                        text=prompt(c['p'],rel,style=style).split('Requested outputs:')[0]+'Solve both problems. Show the calculations you need, then give the final answers labeled A and B.'
                        for decoder in ('full','block32'):
                            result=generate_text(b,text,max_new_tokens=512,tokens_per_step=1) if decoder=='full' else generate_blockwise(b,text,max_new_tokens=512,block_size=32)
                            nfe+=result['nfe'];count+=1
                            record=dict(case=c['id'],relation=rel,style=style,decoder=decoder,p=c['p'],prompt=text,gold=gold(c['p'],rel),**result)
                            f.write(json.dumps(record)+'\n');f.flush();save()
                            print('\nCASE',c['id'],rel,style,decoder,'EOS',result['eos_found'],'TOKENS',result['effective_output_tokens'],flush=True)
                            print(result['text'],flush=True)
                            if time.monotonic()-started>1200:raise TimeoutError('20 minute generation budget')
            manifest['status']='complete'
        except Exception as e:
            manifest.update(status='failed',error=repr(e));raise
        finally:save()
    print('FREE GENERATION COMPLETE',count,'outputs',nfe,'forwards',flush=True)

if __name__=='__main__':main()
