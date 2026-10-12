"""Prompt/layout calibration without supplied intermediate answers."""
import random
import re
from .shared_causal import examples, fill, read_values
from .shared_followup import cases, prepare

VARIANTS = ('old_masked', 'no_example_masked', 'answer_only', 'generated_intermediates', 'natural')


def splits():
    used = {tuple(sorted(p.items())) for row in examples(6,123) for p in (row['base'],row['donor'],row['same'])}
    used.update(tuple(sorted(row[v].items())) for row in cases(10,124) for v in ('base','donor','same'))
    result = {}
    for split, seed, n in [('calibration',125,4),('heldout',126,10)]:
        rng=random.Random(seed); rows=[]
        for suite in ('replication','branches'):
            count=0
            while count<n:
                if suite=='replication':
                    p=dict(x=rng.randint(4,8),y=rng.randint(6,9),u=rng.randint(2,5),v=rng.randint(2,5))
                else:
                    p=dict(x=rng.randint(4,7),y=rng.randint(6,8),u=rng.randint(4,6),r=rng.randint(6,8),v=rng.randint(3,6),s=rng.randint(7,8))
                key=tuple(sorted(p.items()))
                if key in used: continue
                used.add(key)
                rows.append(dict(id=f'{split}-{suite}-{count}',suite=suite,base=p));count+=1
        result[split]=rows
    return result


def make_canvas(b,case,variant):
    if variant=='old_masked': return prepare(b,case,visible=False)
    _,_,oldprompt=prepare(b,case,visible=False)
    prompt=oldprompt.split('\nExample:')[0]
    labels='CAB' if case['suite']=='replication' else 'CUVAB'
    if variant=='answer_only':
        prompt=prompt.split('. Fill')[0]+'. Compute A and B. Fill only the two final integer values in the supplied answer template. No explanation.'
        labels='AB'
    tok=b.tokenizer
    prefix=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
    text=prefix+'; '.join(f'{k} = 00' for k in labels)+'.'
    ids=tok.encode(text,add_special_tokens=False)
    decoded=[tok.decode(ids[:i],skip_special_tokens=False,clean_up_tokenization_spaces=False) for i in range(len(ids)+1)]
    if decoded[-1]!=text: raise ValueError('Tokenizer roundtrip failed')
    spans={}
    for m in re.finditer(r'([CUVAB]) = (00)',text[len(prefix):]):
        lo,hi=len(prefix)+m.start(2),len(prefix)+m.end(2)
        spans[m[1]]=list(range(decoded.index(text[:lo]),decoded.index(text[:hi])))
    if set(spans)!=set(labels): raise ValueError('Missing spans')
    for ps in spans.values():
        for pos in ps: ids[pos]=b.mask_id
    return ids,spans,prompt


def natural_values(text):
    if '<think>' in text and '</think>' not in text: return dict(A=None,B=None)
    text=text.rsplit('</think>',1)[-1]
    matches=list(re.finditer(r'^FINAL: A\s*=\s*(\d+)\s*;\s*B\s*=\s*(\d+)\s*\.?\s*$',text,re.M))
    if len(matches)!=1: return dict(A=None,B=None)
    return dict(A=int(matches[0][1]),B=int(matches[0][2]))


def run_one(b,case,variant):
    if variant not in VARIANTS: raise ValueError(variant)
    if variant=='natural':
        from .text_generation import generate_blockwise
        _,_,old=prepare(b,case,visible=False)
        prompt=old.split('. Fill')[0]+'. Calculate the intermediate values and then A and B. Keep the calculation brief. End with exactly one line in this format: FINAL: A = <integer>; B = <integer>.'
        generation=generate_blockwise(b,prompt,max_new_tokens=768,block_size=32)
        return dict(prompt=prompt,values=natural_values(generation['text']),nfe=generation['nfe'],generation=generation)
    ids,spans,prompt=make_canvas(b,case,variant)
    state=ids.copy();writes=[]
    if variant=='generated_intermediates':
        state,w=fill(b,state,[p for k,ps in spans.items() if k not in 'AB' for p in ps]);writes+=w
    end,w=fill(b,state,spans['A']+spans['B']);writes+=w
    result=read_values(b,end,spans)
    return dict(prompt=prompt,initial_state=ids,spans=spans,ids=end,writes=writes,values=result['values'],texts=result['texts'],nfe=len(writes))


def choose(rows,suite):
    # Frozen order breaks ties: retain hidden slots if equally accurate, then lower cost.
    rank={v:i for i,v in enumerate(VARIANTS)}
    candidates=[v for v in VARIANTS if v!='old_masked']
    return max(candidates,key=lambda v:(sum(r['correct'] for r in rows if r['suite']==suite and r['variant']==v),
                                       -sum(r['nfe'] for r in rows if r['suite']==suite and r['variant']==v),-rank[v]))
