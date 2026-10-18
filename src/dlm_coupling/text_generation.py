"""Full-sequence masked generation with native token IDs and no gold input."""
import math


def generate_text(backend, prompt, max_new_tokens=64, tokens_per_step=1):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    if type(tokens_per_step) is not int or tokens_per_step < 1:
        raise ValueError('tokens_per_step must be a positive integer')
    tokenizer = backend.tokenizer
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                          tokenize=False, add_generation_prompt=True)
    prefix = list(tokenizer.encode(text, add_special_tokens=False))
    if backend.mask_id in prefix: raise ValueError('Prompt contains mask token')
    limits = [getattr(getattr(backend, 'model_config', None), key, None)
              for key in ('max_position_embeddings', 'max_sequence_length', 'max_seq_len')]
    limits = [x for x in limits if type(x) is int and x > 0]
    if limits and len(prefix) + max_new_tokens > min(limits):
        raise ValueError('Prompt plus output budget exceeds context length')
    tokens = prefix + [backend.mask_id] * max_new_tokens
    steps = []
    while backend.mask_id in tokens[len(prefix):]:
        positions = [i for i in range(len(prefix), len(tokens)) if tokens[i] == backend.mask_id]
        predictions = backend.predict(tokens.copy(), positions, [[] for _ in positions])
        if len(predictions) != len(positions): raise ValueError('Missing predictions')
        for pred in predictions:
            if type(pred['token_id']) is not int or pred['token_id'] == backend.mask_id:
                raise ValueError('Invalid predicted token')
            if not math.isfinite(pred['confidence']) or not 0 <= pred['confidence'] <= 1:
                raise ValueError('Invalid confidence')
        selected = sorted(range(len(positions)), key=lambda i: (-predictions[i]['confidence'], positions[i]))[:tokens_per_step]
        writes = []
        for i in selected:
            pos, pred = positions[i], predictions[i]
            tokens[pos] = pred['token_id']
            writes.append({'output_position': pos - len(prefix), 'token_id': pred['token_id'],
                           'text': pred['token_text'], 'confidence': pred['confidence']})
        steps.append({'step': len(steps), 'writes': writes})
    if tokens[:len(prefix)] != prefix: raise AssertionError('Prompt changed')
    output = tokens[len(prefix):]
    eos = getattr(tokenizer, 'eos_token_id', None)
    eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
    end = next((i for i, token in enumerate(output) if token in eos_ids), len(output))
    return {'text': tokenizer.decode(output[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False),
            'raw_text': tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            'output_token_ids': output, 'prompt_tokens': len(prefix), 'output_budget': max_new_tokens,
            'effective_output_tokens': end, 'eos_found': end < len(output), 'nfe': len(steps),
            'steps': steps, 'sampler': 'full-sequence confidence top-k; mask excluded; no numeric restriction',
            'tokens_per_step': tokens_per_step, 'gold_used_for_generation': False}


def generate_blockwise(backend,prompt,max_new_tokens=512,block_size=32,tokens_per_step=1):
    """On-demand blocks with confidence ordering; finish a block before EOS stop.

    Future blocks are appended on demand. This is an explicitly separate
    semi-autoregressive baseline, not the full-sequence parallel experiment.
    """
    if type(max_new_tokens) is not int or max_new_tokens<1: raise ValueError('Invalid budget')
    if type(block_size) is not int or block_size<1: raise ValueError('Invalid block size')
    if type(tokens_per_step) is not int or tokens_per_step<1: raise ValueError('Invalid tokens per step')
    tok=backend.tokenizer
    text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
    prefix=list(tok.encode(text,add_special_tokens=False))
    if backend.mask_id in prefix: raise ValueError('Prompt contains mask')
    limits=[getattr(getattr(backend,'model_config',None),k,None) for k in ('max_position_embeddings','max_sequence_length','max_seq_len')]
    limits=[v for v in limits if type(v) is int and v>0]
    if limits and len(prefix)+max_new_tokens>min(limits): raise ValueError('Context budget exceeded')
    eos=getattr(tok,'eos_token_id',None)
    eos_ids=set(eos if isinstance(eos,list) else [eos])-{None}
    output=[];steps=[]
    while len(output)<max_new_tokens:
        n=min(block_size,max_new_tokens-len(output))
        tokens=prefix+output+[backend.mask_id]*n
        start=len(prefix)+len(output)
        while backend.mask_id in tokens[start:]:
            positions=[i for i in range(start,len(tokens)) if tokens[i]==backend.mask_id]
            preds=backend.predict(tokens,positions,[[] for _ in positions])
            if len(preds)!=len(positions): raise ValueError('Missing predictions')
            if any(p['token_id']==backend.mask_id or not math.isfinite(p['confidence']) or not 0<=p['confidence']<=1 for p in preds):
                raise ValueError('Invalid prediction')
            selected=sorted(range(len(preds)),key=lambda i:(-preds[i]['confidence'],positions[i]))[:tokens_per_step]
            writes=[]
            for i in selected:
                pos,p=positions[i],preds[i];tokens[pos]=p['token_id']
                writes.append({'output_position':pos-len(prefix),'token_id':p['token_id'],
                               'text':p['token_text'],'confidence':p['confidence']})
            steps.append({'step':len(steps),'writes':writes})
        output=tokens[len(prefix):]
        if any(t in eos_ids for t in output): break
    end=next((i for i,t in enumerate(output) if t in eos_ids),len(output))
    return {'text':tok.decode(output[:end],skip_special_tokens=False,clean_up_tokenization_spaces=False),
            'raw_text':tok.decode(output,skip_special_tokens=False,clean_up_tokenization_spaces=False),
            'output_token_ids':output,'prompt_tokens':len(prefix),'output_budget':max_new_tokens,
            'generated_canvas_tokens':len(output),'effective_output_tokens':end,'eos_found':end<len(output),
            'nfe':len(steps),'steps':steps,'sampler':'on-demand blockwise confidence',
            'block_size':block_size,'tokens_per_step':tokens_per_step,'gold_used_for_generation':False}
