"""Prospectively specified reasoning/final-answer protocol for benchmark pilots."""
import re
from . import benchmark_transfer as base

PROTOCOL = 'benchmark-transfer-v2'


def prompt(case, requested, context='full'):
    text = base.prompt(case, requested, context)
    text = text.replace('Give short answers, not explanations.', 'You may reason briefly before answering.')
    old = 'Return exactly one line per requested question, in the requested order, as LABEL: answer.'
    instructions = ('You may reason briefly before the final answer. Finish your reasoning before writing FINAL:. '
                    'After FINAL: write exactly one line per requested question, in the requested order, '
                    'as LABEL: short answer. Do not repeat the questions or add explanations after FINAL:. '
                    'If you use <think> tags, close </think> before FINAL:.')
    return text.replace(old, instructions)


def final_answer(text):
    """No answer mining from reasoning; reject malformed/nested reasoning tags.

    Accept FINAL on its own line after completed reasoning, or native closed
    reasoning followed directly by labeled answers, or a direct labeled reply.
    These alternatives are fixed before running v2, not selected by correctness.
    """
    text = text.strip()
    if '<think>' in text or '</think>' in text:
        if not text.startswith('<think>') or text.count('<think>') != 1 or text.count('</think>') != 1:
            return None, 'unfinished_or_malformed_reasoning'
        text = text.split('</think>', 1)[1].strip()
        route = 'closed_think'
    else:
        route = 'direct'
    markers = list(re.finditer(r'(?m)^\s*FINAL:\s*$', text))
    if len(markers) > 1:
        return None, 'duplicate_final_marker'
    if markers:
        text = text[markers[0].end():].strip()
        route += '+final_marker'
    return text, route


def score(case, text, requested):
    final, route = final_answer(text)
    scored = base.score(case, final or '', requested)
    return dict(**scored, final_text=final, extraction_route=route,
                raw_format_valid=base.parse_answers(text, requested) is not None)


def selected_cases(data, per_cell=3):
    pools = {(b,h): [c for c in data['cases'] if c['benchmark']==b and c['hops']==h]
             for b in ('musique','stepgame') for h in (2,3,4)}
    if per_cell < 1 or any(len(p) < per_cell for p in pools.values()):
        raise ValueError('Invalid per-cell sample count')
    # Interleave families/depths to reduce imbalance if the runtime is interrupted.
    return [pools[(b,h)][i] for i in range(per_cell) for h in (2,3,4)
            for b in ('musique','stepgame')]
