"""Controlled arithmetic interchange pilot; not evidence of compute-once reuse."""
import random
import re


def examples(n=6, seed=123):
    rng = random.Random(seed)
    result = []
    while len(result) < n:
        x, y = rng.randint(4, 8), rng.randint(6, 9)
        u, v = rng.randint(2, 5), rng.randint(2, 5)
        base = dict(x=x, y=y, u=u, v=v)
        if base in [r['base'] for r in result]:
            continue
        donor = dict(x=x+1, y=y, u=u+3, v=v+3)
        same = dict(x=x-1 if y<9 else x+1, y=y+1 if y<9 else y-1, u=u+3, v=v+3)
        result.append(dict(id=len(result), base=base, donor=donor, same=same,
                           gold=answers(base), hybrid=answers(base, x+y+1)))
    return result


def answers(p, c=None):
    c = p['x'] + p['y'] if c is None else c
    return dict(C=c, A=c+p['u'], B=2*c+p['v'])


def canvas(backend, p, c=None):
    """Fixed two-digit interface. Token boundaries audited in full context."""
    tok = backend.tokenizer
    prompt = (f"Compute C = {p['x']} + {p['y']}; A = C + {p['u']}; "
              f"B = 2 * C + {p['v']}. Fill the three integer values in the supplied "
              "answer template. No explanation.\nExample: C = 4 + 6; A = C + 2; "
              "B = 2 * C + 3. Answer: C = 10; A = 12; B = 23.")
    prefix = tok.apply_chat_template([dict(role='user', content=prompt)],
                                     tokenize=False, add_generation_prompt=True)
    text = prefix + 'C = 00; A = 00; B = 00.'
    ids = tok.encode(text, add_special_tokens=False)
    decoded = [tok.decode(ids[:i], skip_special_tokens=False,
                          clean_up_tokenization_spaces=False) for i in range(len(ids)+1)]
    if decoded[-1] != text:
        raise ValueError('Tokenizer does not round-trip full canvas')
    spans = {}
    for m in re.finditer(r'([CAB]) = (00)', text[len(prefix):]):
        lo, hi = len(prefix)+m.start(2), len(prefix)+m.end(2)
        a, b = decoded.index(text[:lo]), decoded.index(text[:hi])
        spans[m[1]] = list(range(a,b))
    for positions in spans.values():
        for pos in positions:
            ids[pos] = backend.mask_id
    if c is not None:
        substitute(backend, ids, spans['C'], c)
    return ids, spans, len(tok.encode(prefix, add_special_tokens=False)), prompt


def substitute(backend, ids, positions, value):
    tok = backend.tokenizer
    tokens = tok.encode(str(value), add_special_tokens=False)
    if len(tokens) != len(positions) or tok.decode(tokens) != str(value):
        raise ValueError('Counterfactual numeral has incompatible token boundaries')
    for pos, token in zip(positions, tokens):
        ids[pos] = token


def fill(backend, ids, positions, first_context=None):
    """Full-vocabulary greedy confidence decoding. Optional one-forward patch."""
    from contextlib import nullcontext
    ids = list(ids)
    writes = []
    while any(ids[p] == backend.mask_id for p in positions):
        missing = [p for p in positions if ids[p] == backend.mask_id]
        with first_context if not writes and first_context is not None else nullcontext():
            preds = backend.predict(ids, missing, [[] for _ in missing])
        j = max(range(len(missing)), key=lambda i:(preds[i]['confidence'], -missing[i]))
        if preds[j]['token_id'] == backend.mask_id:
            raise ValueError('Mask selected')
        ids[missing[j]] = preds[j]['token_id']
        writes.append(dict(position=missing[j], **preds[j],
                           predictions=[dict(position=p, **v) for p,v in zip(missing,preds)]))
    return ids, writes


def read_values(backend, ids, spans):
    texts = {k: backend.tokenizer.decode([ids[p] for p in ps]) for k, ps in spans.items()}
    values = {k: int(s) if re.fullmatch(r'\d{2}', s) else None for k,s in texts.items()}
    return dict(values=values, texts=texts, valid=all(v is not None for v in values.values()))


class ResidualPatch:
    """Module-output patch with exact shape checks and guaranteed hook cleanup."""
    def __init__(self, module, positions, donor=None):
        self.module, self.positions, self.donor = module, positions, donor
        self.captured = None
        self.calls = 0

    def __enter__(self):
        def hook(module, args, output):
            self.calls += 1
            h = output[0] if isinstance(output, tuple) else output
            if h.ndim != 3 or h.shape[0] != 1:
                raise ValueError('Unexpected residual shape')
            self.captured = h[:, self.positions, :].detach().clone()
            if self.donor is None:
                return None
            if self.donor.shape != self.captured.shape:
                raise ValueError('Patch shape mismatch')
            changed = h.clone()
            changed[:, self.positions, :] = self.donor.to(h)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        self.handle.remove()
        if args[0] is None and self.calls != 1:
            raise AssertionError(f'Expected one patched forward, got {self.calls}')
