"""Self-contained ordinary-integer rendering of official iGSM subgraphs."""
import hashlib
import json
import re
from .igsm import ancestors, evaluate_graph, relation, references


def expression_text(e, names):
    if 'literal' in e: return str(e['literal'])
    if 'ref' in e: return names[e['ref']]
    op={'sum':' + ', 'diff':' - ', 'mul':' * '}[e['op']]
    return '('+op.join(expression_text(a,names) for a in e['args'])+')'


def facts_text(nodes):
    names={key:f'Q{i+1}' for i,key in enumerate(nodes)}
    return '\n'.join(names[k]+' = '+expression_text(n['expression'],names) for k,n in nodes.items())


def validate_pair(pair):
    nodes=pair['nodes']
    values=evaluate_graph(nodes,modulus=None)
    assert pair['arithmetic']=='ordinary_integer'
    assert pair['facts']==facts_text(nodes)
    assert pair['facts_sha256']==hashlib.sha256(pair['facts'].encode()).hexdigest()
    assert all(type(v) is int and 0<=v<=99 for v in values.values())
    a,b=(pair['targets'][q]['node'] for q in 'AB')
    assert relation(nodes,a,b)==pair['relation']
    for q in 'AB':
        assert pair['targets'][q]['gold']==values[pair['targets'][q]['node']]
    needed=ancestors(nodes,a)|ancestors(nodes,b)|{a,b}
    assert set(nodes)==needed


def render_v1(pair,order):
    if order not in ('A','B','AB','BA'): raise ValueError('Invalid order')
    names={key:f'Q{i+1}' for i,key in enumerate(pair['nodes'])}
    lines=['Solve this arithmetic exercise. Use ordinary integer arithmetic.',
           'Each Q symbol is one quantity. Every quantity is explicitly defined below.',
           'An equals sign defines a value; * means multiplication. Follow the definitions exactly.',
           'Definitions:',pair['facts'],'Questions:']
    lines += [q+': What is '+names[pair['targets'][q]['node']]+'?' for q in order]
    lines += ['Calculate briefly, then give the final answer on the last line.',
              'The last line must start with FINAL_JSON: followed by a JSON object with exactly the keys '+
              ', '.join('"'+q+'"' for q in order)+' and their integer values.',
              'If you use a <think> block, close it before the final answer.']
    return '\n'.join(lines)


def render(pair,order,version='v2'):
    if version=='v1': return render_v1(pair,order)
    if version!='v2' or order not in ('A','B','AB','BA'): raise ValueError('Invalid prompt version or order')
    names={key:f'Q{i+1}' for i,key in enumerate(pair['nodes'])}
    lines=['Solve the arithmetic exercise using ordinary integers.',
           'Every variable is explicitly defined. Use all terms in each expression; * means multiplication.',
           'For each requested quantity, substitute the numbers and briefly show the calculation.',
           'End with one line: FINAL_JSON: followed by a JSON object containing only the requested question labels and integer answers.',
           '', 'Example (a different exercise):',
           'R1 = 3', 'R2 = 4', 'R3 = R1 + R2', 'R4 = R3 + 2', 'R5 = 2 * R3',
           'Questions: A asks for R4; B asks for R5.',
           'Calculation: R3 = 3 + 4 = 7; R4 = 7 + 2 = 9; R5 = 2 * 7 = 14.',
           'FINAL_JSON: {"A": 9, "B": 14}',
           '', 'Now solve this exercise:', 'Definitions:',pair['facts'],'Questions:']
    lines += [q+': What is '+names[pair['targets'][q]['node']]+'?' for q in order]
    lines += ['Calculate the requested quantities, then end with FINAL_JSON: using exactly the keys '+
              ', '.join('"'+q+'"' for q in order)+'.']
    return '\n'.join(lines)


def parse_answers(text,labels):
    # Never score a guessed answer inside unfinished reasoning.
    if text.count('<think>') != text.count('</think>'): return {},'unfinished_reasoning'
    if 'FINAL_JSON:' in text: text=text.rsplit('FINAL_JSON:',1)[1]
    text=text.strip()
    if text.startswith('```'):
        match=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',text,re.DOTALL)
        if not match: return {},'format'
        text=match.group(1)
    def unique(items):
        obj={}
        for k,v in items:
            if k in obj: raise ValueError('Duplicate key')
            obj[k]=v
        return obj
    try: obj=json.loads(text,object_pairs_hook=unique)
    except (ValueError,TypeError): return {},'format'
    if not isinstance(obj,dict) or set(obj)!=set(labels): return {},'keys'
    if any(type(v) is not int for v in obj.values()): return {},'domain'
    return obj,None
