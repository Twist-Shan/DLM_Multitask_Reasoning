"""Exploratory residual interchange at naturally generated arithmetic spans.

Replay only committed tokens, never future completed text. Layer selection and
rollouts use the same two diagnostic groups: no held-out causal success claim.
"""
import hashlib
import json
import math
import os
import re
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dlm_coupling.backends.hf import HFBackend
from dlm_coupling.implicit_sharing import prompt, sources, gold
from dlm_coupling.shared_causal import ResidualPatch
from dlm_coupling.text_generation import generate_blockwise

LAYERS = (0, 3, 7, 11, 15, 19, 23, 27, 30)


def spans(tok, row):
    """Locate the first computed common sum and its first explicit B reuse."""
    text = row['text']
    c = row['p']['x'] + row['p']['y']
    first = re.search(r'x\s*\+\s*y\s*=\s*(?:\d+\s*\+\s*\d+\s*=\s*)?(\d+)', text)
    if first is None or int(first[1]) != c:
        raise ValueError('No unambiguous first computed x+y span')
    boundary=text.find('For B:',first.end())
    if boundary<0: raise ValueError('No explicit B section boundary')
    reuse = re.search(r'x\s*\+\s*y\s*(?:=|is)\s*(\d+)', text[boundary:], re.I)
    if reuse is None:
        reuse=re.search(r'2\s*\*\s*(\d+)',text[boundary:])
    if reuse is None or int(reuse[1]) != c:
        raise ValueError('No explicit natural B reuse span')
    ids = row['output_token_ids']
    dec = [tok.decode(ids[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in range(len(ids)+1)]
    def locate(lo, hi):
        ps = [i for i in range(len(ids)) if len(dec[i]) < hi and len(dec[i+1]) > lo]
        if not ps or dec[ps[0]] != text[:lo] or dec[ps[-1]+1] != text[:hi]:
            raise ValueError('Numeral does not align exactly to original token boundaries')
        return ps
    common = locate(first.start(1), first.end(1))
    reused = locate(boundary+reuse.start(1), boundary+reuse.end(1))
    # All digits must still be uncommitted at this event; earliest reuse write.
    step = min(s['step'] for s in row['steps'] for w in s['writes'] if w['output_position'] in reused)
    return common, reused, step


def replay(b, row, stop):
    tok = b.tokenizer
    pre = tok.encode(tok.apply_chat_template([dict(role='user', content=row['prompt'])], tokenize=False, add_generation_prompt=True), add_special_tokens=False)
    writes = [w for s in row['steps'][:stop] for w in s['writes']]
    next_pos = row['steps'][stop]['writes'][0]['output_position']
    n = (next_pos//32+1)*32
    ids = pre + [b.mask_id]*n
    for w in writes:
        ids[len(pre)+w['output_position']] = w['token_id']
    return ids, len(pre)


def resume(b, ids, plen, layer=None, positions=None, values=None, mode='none'):
    """Continue from exact saved mask state; a patch lasts one forward or block."""
    ids = ids.copy(); initial_end = len(ids); nfe = 0; writes = []
    eos = b.tokenizer.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) - {None}
    while len(ids)-plen <= 768:
        missing = [i for i in range(plen, len(ids)) if ids[i] == b.mask_id]
        if not missing:
            if any(t in eos for t in ids[plen:]) or len(ids)-plen == 768: break
            ids += [b.mask_id]*32
            missing = list(range(len(ids)-32, len(ids)))
        active = layer is not None and ((mode == 'single' and nfe == 0) or (mode == 'block' and len(ids) == initial_end))
        with ExitStack() as stack:
            if active: stack.enter_context(ResidualPatch(b._model.model.layers[layer], positions, values))
            preds = b.predict(ids, missing, [[] for _ in missing])
        j = max(range(len(missing)), key=lambda i: (preds[i]['confidence'], -missing[i]))
        ids[missing[j]] = preds[j]['token_id']; nfe += 1
        writes.append(dict(output_position=missing[j]-plen, token_id=preds[j]['token_id'], patched=active))
    output = ids[plen:]
    end = next((i for i,t in enumerate(output) if t in eos), len(output))
    return dict(text=b.tokenizer.decode(output[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                output_token_ids=output, writes=writes, nfe=nfe, eos_found=end<len(output))


def main():
    out = ROOT/'runs'/os.environ.get('DLM_PATCH_RUN_NAME','free-reasoning-patching'); out.mkdir(parents=True, exist_ok=False)
    cached=[]
    if os.environ.get('DLM_PATCH_REUSE'):
        cached=[json.loads(s) for s in (ROOT/'runs'/os.environ['DLM_PATCH_REUSE']/'results.jsonl').read_text().splitlines()]
    prior = ROOT/'runs/free-reasoning'
    manifest = json.loads((prior/'manifest.json').read_text())
    rows = [json.loads(s) for s in (prior/'results.jsonl').read_text().splitlines()]
    continued = prior/'continuations.jsonl'
    if continued.exists():
        rows.extend(json.loads(s) for s in continued.read_text().splitlines())
    # A continuation replaces its original trajectory, rather than adding a sample.
    latest = {r['case']:r for r in rows if r['decoder']=='block32' and r['style']=='symbolic' and r['relation']=='shared'}
    targets = list(latest.values())
    assert len(targets)==2
    assert all(r['eos_found'] for r in targets), 'Complete Target trajectories before patching'
    b = HFBackend(manifest['model'], load_model=True); b._torch.manual_seed(123)
    started = time.monotonic()
    meta = dict(status='running', seed=123, model=manifest['model'], backend=b.metadata, layers_one_based=[l+1 for l in LAYERS],
                selection='Two reused symbolic cases; exploratory layer selection on the same cases; no held-out evaluation',
                source='Natural free generation on changed facts; semantic span alignment, separate natural commitment event',
                source_controls=['self', 'diff_C_same_answers', 'same_C_diff_answers'],
                scope='Block32 semi-autoregressive decoder; residual module output; no claim specific to full-sequence DLM',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    meta['reused_run']=os.environ.get('DLM_PATCH_REUSE')
    meta['span_rule']='First x+y numeric result; first B-section x+y numeric mention, otherwise 2*C multiplier operand; manually inspect extracted spans'
    def save():
        meta['elapsed_seconds']=time.monotonic()-started
        (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    def emit(row):
        with (out/'results.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
    save()
    try:
        for target in targets:
            case=target['case']; cp,rp,step=spans(b.tokenizer,target)
            state,plen=replay(b,target,step); common=[plen+i for i in cp]; query=plen+rp[-1]
            assert all(state[p]!=b.mask_id for p in common)
            assert all(state[plen+p]==b.mask_id for p in rp)
            c=target['p']['x']+target['p']['y']; tid=b.tokenizer.encode(str(c),add_special_tokens=False); sid=b.tokenizer.encode(str(c+1),add_special_tokens=False)
            assert len(tid)==len(sid)==len(cp) and tid[:-1]==sid[:-1] and tid[-1]!=sid[-1]
            probes=[tid[-1],sid[-1]]
            def measure(ids):
                r=b.predict(ids,[query],[probes])[0]
                p=r['probabilities']; r['log_odds']=math.log(max(p[str(sid[-1])],1e-38))-math.log(max(p[str(tid[-1])],1e-38))
                return r
            base=measure(state)
            emit(dict(kind='event',case=case,prompt=target['prompt'],p=target['p'],common_positions=common,reuse_positions=[plen+i for i in rp],step=step,state=state,prompt_tokens=plen,baseline=base,target_gold=gold(target['p']),counterfactual_B=2*(c+1)+target['p']['v']+target['p']['s']))
            print('EVENT',case,'step',step,'common',cp,'reuse',rp,'baseline',base['token_text'],flush=True)
            caps={}
            with ExitStack() as stack:
                hh=[stack.enter_context(ResidualPatch(b._model.model.layers[l],common)) for l in LAYERS]
                measure(state)
            for l,h in zip(LAYERS,hh): caps[l]=h.captured
            for l in LAYERS:
                with ResidualPatch(b._model.model.layers[l],common,caps[l]): test=measure(state)
                assert test==base,'Self patch differs'
                emit(dict(kind='scan',case=case,source_kind='self',site='common',layer=l+1,delta=0,**test))
            # Resume untouched exact replay and verify deterministic final output.
            prior_rr=next((r for r in cached if r['kind']=='rollout' and r['case']==case and r['condition']=='baseline'),None)
            rr=prior_rr or resume(b,state,plen)
            assert rr['text']==target['text'],'Replay baseline diverged from original text'
            emit(dict(rr,kind='rollout',case=case,condition='baseline',reused=prior_rr is not None))
            changed=state.copy()
            for pos,t in zip(common,sid): changed[pos]=t
            prior_rr=next((r for r in cached if r['kind']=='rollout' and r['case']==case and r['condition']=='token_common_plus_one'),None)
            rr=prior_rr or resume(b,changed,plen)
            emit(dict(rr,kind='rollout',case=case,condition='token_common_plus_one',reused=prior_rr is not None))
            scans=[]; source_caps={}
            for sk in ('diff_C_same_answers','same_C_diff_answers'):
                p=sources(target['p'],'shared')[sk]
                user=prompt(p,'shared',style='symbolic').split('Requested outputs:')[0]+'Solve both problems. Show the calculations you need, then give the final answers labeled A and B.'
                prior_source=next((r for r in cached if r['kind']=='source_generation' and r['case']==case and r['source_kind']==sk),None)
                source=prior_source or dict(case=case,source_kind=sk,p=p,prompt=user,**generate_blockwise(b,user,max_new_tokens=768,block_size=32))
                emit(dict(source,kind='source_generation',reused=prior_source is not None))
                print('SOURCE',case,sk,source['text'],flush=True)
                try: sc,sr,ss=spans(b.tokenizer,source)
                except ValueError as e:
                    emit(dict(kind='skipped_source',case=case,source_kind=sk,reason=str(e)));continue
                source_state,sp=replay(b,source,ss); source_common=[sp+i for i in sc]
                assert len(source_common)==len(common)
                assert all(source_state[i]!=b.mask_id for i in source_common)
                with ExitStack() as stack:
                    hh=[stack.enter_context(ResidualPatch(b._model.model.layers[l],source_common)) for l in LAYERS]
                    b.predict(source_state,[sp+sr[-1]],[[]])
                values={l:h.captured for l,h in zip(LAYERS,hh)}
                source_caps[sk]=values
                emit(dict(kind='source_event',case=case,source_kind=sk,step=ss,state=source_state,common_positions=source_common,source_gold=gold(p)))
                # Same source vector at unrelated earlier text positions, matched span width.
                control=[plen+i for i in range(4,4+len(common))]
                assert all(state[i]!=b.mask_id for i in control)
                for site,positions in [('common',common),('unrelated_text',control)]:
                    for l in LAYERS:
                        with ResidualPatch(b._model.model.layers[l],positions,values[l]): m=measure(state)
                        record=dict(kind='scan',case=case,source_kind=sk,site=site,layer=l+1,delta=m['log_odds']-base['log_odds'],**m)
                        emit(record); scans.append(record)
            candidates=[r for r in scans if r['source_kind']=='diff_C_same_answers' and r['site']=='common']
            if candidates:
                best=max(candidates,key=lambda r:(r['delta'],-r['layer'])); l=best['layer']-1
                emit(dict(kind='selection',case=case,criterion='maximum local delta, exploratory',selected=best))
                for mode in ('single','block'):
                    rr=resume(b,state,plen,l,common,source_caps['diff_C_same_answers'][l],mode)
                    emit(dict(kind='rollout',case=case,condition='activation_'+mode,layer=l+1,**rr))
                    print('ROLLOUT',case,mode,rr['text'],flush=True)
            print('CASE COMPLETE',case,flush=True);save()
            if time.monotonic()-started>1200: raise TimeoutError('20 minute budget')
        meta['status']='complete'
    except Exception as exc:
        meta.update(status='failed',error=repr(exc));raise
    finally: save()
    print('PATCHING COMPLETE',flush=True)


if __name__=='__main__':main()
