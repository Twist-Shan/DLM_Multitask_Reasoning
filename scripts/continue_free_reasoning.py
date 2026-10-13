"""Continue only truncated block32 outputs, preserving their original 512 tokens."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dlm_coupling.backends.hf import HFBackend

def main():
    out=ROOT/'runs/free-reasoning'
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='complete'
    rows=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    pending=[r for r in rows if r['decoder']=='block32' and not r['eos_found']]
    if not pending: print('NO TRUNCATED BLOCK OUTPUTS');return
    b=HFBackend(manifest['model'],load_model=True);b._torch.manual_seed(123);tok=b.tokenizer
    eos=tok.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])-{None}
    with (out/'continuations.jsonl').open('x',encoding='utf-8') as f:
        for original in pending:
            r=dict(original);output=r['output_token_ids'].copy();steps=r['steps'].copy();initial=len(output)
            text=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True)
            prefix=tok.encode(text,add_special_tokens=False)
            while len(output)<1024:
                tokens=prefix+output+[b.mask_id]*32;start=len(prefix)+len(output)
                for _ in range(32):
                    positions=[i for i in range(start,len(tokens)) if tokens[i]==b.mask_id]
                    pp=b.predict(tokens,positions,[[] for _ in positions])
                    j=max(range(len(pp)),key=lambda i:(pp[i]['confidence'],-positions[i]))
                    p=pp[j];pos=positions[j];tokens[pos]=p['token_id']
                    steps.append(dict(step=len(steps),writes=[dict(output_position=pos-len(prefix),token_id=p['token_id'],text=p['token_text'],confidence=p['confidence'])]))
                output=tokens[len(prefix):]
                if any(t in eos for t in output):break
            assert output[:initial]==original['output_token_ids']
            end=next((i for i,t in enumerate(output) if t in eos),len(output))
            r.update(output_token_ids=output,steps=steps,output_budget=1024,generated_canvas_tokens=len(output),
                effective_output_tokens=end,eos_found=end<len(output),nfe=len(steps),additional_nfe=len(steps)-original['nfe'],
                text=tok.decode(output[:end],skip_special_tokens=False,clean_up_tokenization_spaces=False),
                raw_text=tok.decode(output,skip_special_tokens=False,clean_up_tokenization_spaces=False),continued_from_budget=512)
            f.write(json.dumps(r)+'\n');f.flush()
            print('CONTINUED',r['case'],r['relation'],r['style'],'EOS',r['eos_found'],'TOKENS',end,flush=True)
            print(r['text'],flush=True)
    print('CONTINUATIONS COMPLETE',len(pending),flush=True)

if __name__=='__main__':main()
