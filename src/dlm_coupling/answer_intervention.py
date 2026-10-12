"""Late answer-token interventions; no claim about hidden-state mechanisms."""
import re
from .tasks.igsm_explicit import parse_answers


def answer_events(tokenizer,row):
    text=row['trace']['text'];answers,error=parse_answers(text,row['order'])
    if error: return {'error':error,'answers':{}}
    start=text.rfind('FINAL_JSON:')
    start=start+len('FINAL_JSON:') if start>=0 else 0
    ids=row['trace']['output_token_ids'][:row['trace']['effective_output_tokens']]
    prefixes=[tokenizer.decode(ids[:i],skip_special_tokens=False,clean_up_tokenization_spaces=False) for i in range(len(ids)+1)]
    if prefixes[-1]!=text: raise ValueError('Decoded text mismatch')
    writes={w['output_position']:s['step'] for s in row['trace']['steps'] for w in s['writes']}
    result={}
    for m in re.finditer(r'"([AB])"\s*:\s*(-?\d+)',text[start:]):
        label=m[1];lo=start+m.start(2);hi=start+m.end(2)
        # Exact token boundaries: never overwrite adjacent syntax or whitespace.
        left=[i for i,p in enumerate(prefixes) if len(p)==lo and p==text[:lo]]
        right=[i for i,p in enumerate(prefixes) if len(p)==hi and p==text[:hi]]
        if not left or not right: continue
        a,b=left[-1],right[-1];positions=list(range(a,b))
        if not positions or int(m[2])!=answers[label]: continue
        result[label]={'value':int(m[2]),'positions':positions,'token_ids':ids[a:b],
                       'start_step':min(writes[p] for p in positions),
                       'complete_step':max(writes[p] for p in positions),
                       'blocks':sorted({p//row['trace']['block_size'] for p in positions})}
    return {'error':None if len(result)==len(answers) else 'nonexact_token_boundary','answers':result}


def choose_fork(tokenizer,row,events,source,target):
    a=events['answers'].get(source);b=events['answers'].get(target)
    if a is None or b is None: return None,'missing_span'
    if a['blocks']!=b['blocks'] or len(a['blocks'])!=1: return None,'different_blocks'
    if a['complete_step']>=b['start_step']: return None,'target_already_started'
    old=a['value']
    for value in (old+1,old-1):
        if value<0 or len(str(value))!=len(str(old)): continue
        tokens=tokenizer.encode(str(value),add_special_tokens=False)
        if len(tokens)!=len(a['positions']): continue
        if tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=str(value): continue
        return {'source':source,'target':target,'cut_step':a['complete_step'],
                'positions':a['positions'],'old_value':old,'new_value':value,
                'old_token_ids':a['token_ids'],'new_token_ids':tokens,
                'target_positions':b['positions']},None
    return None,'no_equal_width_foil'


def replay_state(backend,row,fork,changed=False):
    t=row['trace'];size=t['block_size']
    rendered=backend.tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True)
    prefix=list(backend.tokenizer.encode(rendered,add_special_tokens=False))
    assert len(prefix)==t['prompt_tokens']
    steps=t['steps'][:fork['cut_step']+1]
    count=(max(w['output_position'] for s in steps for w in s['writes'])//size+1)*size
    output=[backend.mask_id]*count
    for step in steps:
        for w in step['writes']: output[w['output_position']]=w['token_id']
    assert [output[p] for p in fork['positions']]==fork['old_token_ids']
    assert all(output[p]==backend.mask_id for p in fork['target_positions'])
    if changed:
        for p,v in zip(fork['positions'],fork['new_token_ids']): output[p]=v
    return prefix,output


def continue_fork(backend,row,fork,changed=False):
    prefix,output=replay_state(backend,row,fork,changed)
    t=row['trace'];size=t['block_size'];budget=t['output_budget'];steps=[]
    eos=backend.tokenizer.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])-{None}
    while True:
        positions=[len(prefix)+i for i,v in enumerate(output) if v==backend.mask_id]
        if not positions:
            if any(v in eos for v in output) or len(output)>=budget: break
            output += [backend.mask_id]*min(size,budget-len(output));continue
        preds=backend.predict(prefix+output,positions,[[] for _ in positions])
        i=max(range(len(preds)),key=lambda i:(preds[i]['confidence'],-positions[i]))
        p=positions[i]-len(prefix);v=preds[i]
        if v['token_id']==backend.mask_id: raise ValueError('Predicted mask')
        output[p]=v['token_id']
        steps.append({'output_position':p,'token_id':v['token_id'],'confidence':v['confidence']})
    end=next((i for i,v in enumerate(output) if v in eos),len(output))
    text=backend.tokenizer.decode(output[:end],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    answers,error=parse_answers(text,row['order'])
    return {'text':text,'output_token_ids':output,'answers':answers,'parse_error':error,
            'eos_found':end<len(output),'nfe':len(steps),'writes':steps}
