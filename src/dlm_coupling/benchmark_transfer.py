"""Exploratory benchmark adaptation. No model-based sample selection."""
import ast
from collections import Counter, deque
import hashlib
import json
import random
import re
import string

PROTOCOL = 'benchmark-transfer-v1'
VECTORS = {'left': (-1, 0), 'right': (1, 0), 'over': (0, 1),
           'below': (0, -1), 'upperleft': (-1, 1), 'upperright': (1, 1),
           'lowerleft': (-1, -1), 'lowerright': (1, -1)}
LABELS = {(-1, 0): 'left', (1, 0): 'right', (0, 1): 'above',
          (0, -1): 'below', (-1, 1): 'upper-left', (1, 1): 'upper-right',
          (-1, -1): 'lower-left', (1, -1): 'lower-right', (0, 0): 'overlap'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_score(prediction, answers):
    """MuSiQue/SQuAD answer EM and token F1, best over annotated aliases."""
    p = normalize(prediction).split()
    scores = []
    for answer in answers:
        a = normalize(answer).split()
        overlap = sum((Counter(p) & Counter(a)).values())
        f1 = (2 * overlap / (len(p) + len(a))) if p and a else float(p == a)
        scores.append((float(p == a), f1))
    return dict(em=max(x[0] for x in scores), f1=max(x[1] for x in scores))


def parse_answers(text, requested):
    """Strict one-field-per-line format; duplicate or missing labels fail."""
    fields = {}
    for line in text.strip().splitlines():
        match = re.fullmatch(r'\s*([ABR]):\s*(\S.*?)\s*', line)
        if not match or match[1] in fields:
            return None
        fields[match[1]] = match[2]
    return fields if list(fields) == list(requested) else None


def load_mapping(path):
    """Read published mapping as literal data, never execute downloaded Python."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    mapping = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict))
    return [(re.compile(pattern), replacement) for pattern, replacement in mapping.items()]


def relation(sentence, mapping):
    matches = {m.expand(value) for pattern, value in mapping
               if (m := pattern.fullmatch(sentence)) is not None}
    if len(matches) != 1:
        raise ValueError('unparsed_or_ambiguous_template')
    return next(iter(matches)).split('_')


def direction(dx, dy):
    return LABELS[((dx > 0) - (dx < 0), (dy > 0) - (dy < 0))]


def spatial_audit(row, mapping):
    graph = {}
    for sentence in row['story']:
        a, rel, b = relation(sentence, mapping)
        if rel not in VECTORS:
            raise ValueError('unexpected_relation')
        dx, dy = VECTORS[rel]  # coordinate(a) - coordinate(b)
        graph.setdefault(b, []).append((a, dx, dy))
        graph.setdefault(a, []).append((b, -dx, -dy))
    a, rel, b = relation(row['question'], mapping)
    if rel != 'query' or a not in graph or b not in graph:
        raise ValueError('invalid_query')
    positions, parents = {b: (0, 0)}, {b: None}
    queue = deque([b])
    while queue:
        u = queue.popleft()
        for v, dx, dy in graph[u]:
            pos = (positions[u][0] + dx, positions[u][1] + dy)
            if v in positions and positions[v] != pos:
                raise ValueError('inconsistent_coordinates')
            if v not in positions:
                positions[v], parents[v] = pos, u
                queue.append(v)
    if a not in positions:
        raise ValueError('disconnected_query')
    calculated = direction(*positions[a])
    if calculated != row['label']:
        raise ValueError('published_label_mismatch')
    path, u = [a], a
    while parents[u] is not None:
        u = parents[u]
        path.append(u)
    # A asks about ONE original edge, preserving magnitude information implicitly.
    u, v = path[0:2]
    av = direction(positions[u][0] - positions[v][0], positions[u][1] - positions[v][1])
    return dict(path=path, actual_hops=len(path)-1, declared_k=int(row['k_hop']),
                A_question=f'What is the relation of the agent {u} to the agent {v}?',
                A_answer=av, positions=positions)


def root_question(text):
    if '#' in text:
        raise ValueError('root_contains_reference')
    if ' >> ' in text:
        entity, attribute = text.split(' >> ', 1)
        return f'What is the {attribute} of {entity}?'
    return text


def musique_case(row):
    steps = row['question_decomposition']
    ancestors = set()
    def visit(i):
        if i in ancestors:
            return
        ancestors.add(i)
        for ref in re.findall(r'#(\d+)', steps[i]['question']):
            j = int(ref) - 1
            if not 0 <= j < i:
                raise ValueError('invalid_dependency')
            visit(j)
    visit(len(steps) - 1)
    roots = [i for i in sorted(ancestors) if not re.search(r'#\d+', steps[i]['question'])]
    if not roots or len(ancestors) < 2 or not row['answerable']:
        raise ValueError('invalid_multihop')
    root = roots[0]
    return dict(id=row['id'], benchmark='musique', hops=len(steps),
                graph=row['id'].split('__')[0], original=row,
                questions={'A': root_question(steps[root]['question']), 'B': row['question']},
                answers={'A': [steps[root]['answer']], 'B': [row['answer'], *row['answer_aliases']]},
                root_step=root, dependency_edges=[(int(ref) - 1, i) for i, s in enumerate(steps)
                    for ref in re.findall(r'#(\d+)', s['question'])])


def prepare(musique, stepgame, mapping, per_hop=10, seed=123):
    rng = random.Random(seed)
    cases, exclusions = [], Counter()
    pools = {}
    for row in musique:
        try:
            case = musique_case(row)
            pools.setdefault(('musique', case['hops']), []).append(case)
        except ValueError as exc:
            exclusions['musique:' + str(exc)] += 1
    for i, row in enumerate(stepgame):
        try:
            audit = spatial_audit(row, mapping)
            case = dict(id=f'stepgame-valid-{i}', benchmark='stepgame', hops=audit['actual_hops'],
                        original=row, audit=audit,
                        questions={'A': audit['A_question'], 'B': row['question']},
                        answers={'A': [audit['A_answer']], 'B': [row['label']]})
            pools.setdefault(('stepgame', case['hops']), []).append(case)
        except ValueError as exc:
            exclusions['stepgame:' + str(exc)] += 1
    for benchmark in ('musique', 'stepgame'):
        for hops in (2, 3, 4):
            pool = pools[(benchmark, hops)]
            if len(pool) < per_hop:
                raise ValueError('Insufficient audited cases')
            cases.extend(rng.sample(pool, per_hop))
    # All B / AB / BA / RB conditions receive IDENTICAL context, including R facts.
    for case in cases:
        if case['benchmark'] == 'musique':
            titles = {p['title'] for p in case['original']['paragraphs']}
            others = list(pools[('musique', 2)])
            rng.shuffle(others)
            for other in others:
                if other['id'] == case['id']:
                    continue
                step = other['original']['question_decomposition'][other['root_step']]
                paragraph = next(p for p in other['original']['paragraphs'] if p['idx'] == step['paragraph_support_idx'])
                target_text = json.dumps(case['original']).lower()
                if paragraph['title'] in titles or step['answer'].lower() in target_text:
                    continue
                if any(x.lower() in paragraph['paragraph_text'].lower() for x in case['answers']['B']):
                    continue
                case['R_paragraph'], case['R_source_id'] = paragraph, other['id']
                case['questions']['R'] = other['questions']['A']
                case['answers']['R'] = other['answers']['A']
                break
            else:
                raise ValueError('No disjoint R candidate')
        else:
            used = set(re.findall(r'\b[A-Z]\b', ' '.join(case['original']['story'])))
            free = [x for x in string.ascii_uppercase if x not in used]
            if len(free) < 2:
                raise ValueError('No unused entity labels')
            u, v = free[:2]
            r = rng.choice(['left', 'right', 'above', 'below'])
            case['R_fact'] = f'{u} is {"to the " if r in ("left", "right") else ""}{r}{" of" if r in ("left", "right") else ""} {v}.'
            case['questions']['R'] = f'What is the relation of the agent {u} to the agent {v}?'
            case['answers']['R'] = [r]
    return dict(protocol=PROTOCOL, seed=seed, per_hop=per_hop, cases=cases,
                audit=dict(exclusions=dict(exclusions), eligible={f'{b}:{h}': len(v) for (b,h),v in pools.items()}))


def prompt(case, requested='B', context='full'):
    if requested not in ('B', 'AB', 'BA', 'RB') or context not in ('full', 'support'):
        raise ValueError('Unknown condition')
    if case['benchmark'] == 'musique':
        paragraphs = case['original']['paragraphs']
        if context == 'support':
            paragraphs = [p for p in paragraphs if p['is_supporting']]
        facts = '\n\n'.join(f"[{p['title']}] {p['paragraph_text']}" for p in [*paragraphs, case['R_paragraph']])
        instruction = 'Use only the supplied passages. Give short answers, not explanations.'
    else:
        if context != 'full':
            raise ValueError('Support context applies only to MuSiQue')
        facts = '\n'.join([*case['original']['story'], case['R_fact']])
        instruction = ('Use the supplied spatial facts. Adjacent relations represent unit steps on a grid. '
                       'Answer using exactly one of: left, right, above, below, upper-left, upper-right, '
                       'lower-left, lower-right, overlap.')
    questions = '\n'.join(f'{key}: {case["questions"][key]}' for key in requested)
    return (instruction + '\nReturn exactly one line per requested question, in the requested order, '
            'as LABEL: answer.\n\nContext:\n' + facts + '\n\nQuestions:\n' + questions)


def score(case, text, requested):
    parsed = parse_answers(text, requested)
    scores = {}
    for key in requested:
        if parsed is None:
            scores[key] = dict(em=0., f1=0.)
        elif case['benchmark'] == 'stepgame':
            em = float(parsed[key].lower() == case['answers'][key][0])
            scores[key] = dict(em=em, f1=em)
        else:
            scores[key] = answer_score(parsed[key], case['answers'][key])
    return dict(format_valid=parsed is not None, parsed=parsed, scores=scores)
