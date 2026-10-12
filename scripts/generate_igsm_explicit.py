"""Select short ordinary-integer subgraphs before observing model scores."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from generate_igsm_pairs import serialize,node_id,SHA
from dlm_coupling.tasks.igsm import ancestors,evaluate_graph,operation_count,relation,references
from dlm_coupling.tasks.igsm_explicit import facts_text,validate_pair,render


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split',choices=['calibration','heldout','screen'],required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int)
    ap.add_argument('--per-relation',type=int)
    ap.add_argument('--max-attempts',type=int,default=3000)
    ap.add_argument('--exclude',action='append',default=[])
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists(): raise ValueError('Refusing to overwrite')
    upstream=ROOT/'data/upstream/igsm'
    assert subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()==SHA
    assert not subprocess.check_output(['git','-C',str(upstream),'status','--porcelain','--untracked-files=no'],text=True).strip()
    sys.path.insert(0,str(upstream))
    from data_gen.pretrain.id_gen import IdGen
    from tools.tools import fix_seed
    seed=args.seed if args.seed is not None else 123
    target=args.per_relation if args.per_relation is not None else {'calibration':1,'heldout':2,'screen':10}[args.split]
    if target<1: raise ValueError('per-relation must be positive')
    seen={json.loads(line)['facts_sha256'] for name in args.exclude for line in Path(name).read_text(encoding='utf-8').splitlines()}
    fix_seed(seed)
    counts={r:0 for r in ('independent','shared','downstream')}
    pairs=[]
    for attempt in range(args.max_attempts):
        g=IdGen(max_op=10,max_edge=18,op=6,perm_level=5,detail_level=0)
        g.gen_prob(list(range(23)),p_format='pq')
        p=g.problem
        nodes={node_id(n):{'name':p.get_ntn(n),'expression':serialize(p.sketch[n])}
               for n in p.template.nodes if n[0]!=-1}
        values=evaluate_graph(nodes,modulus=None)
        closures={n:ancestors(nodes,n)|{n} for n in nodes}
        ops={n:sum(operation_count(nodes[k]['expression']) for k in c) for n,c in closures.items()}
        candidates=[n for n,c in closures.items() if 1<=ops[n]<=3
                    and operation_count(nodes[n]['expression'])>=1
                    and all(0<=values[k]<=99 for k in c)]
        choices=[]
        for a in candidates:
            for b in candidates:
                if a==b or values[a]==values[b]: continue
                r=relation(nodes,a,b)
                if r in counts and counts[r]<target:
                    choices.append((ops[a]+ops[b],len(closures[a]|closures[b]),a,b,r))
        if not choices: continue
        _,_,a,b,r=min(choices)
        needed=closures[a]|closures[b]
        # Topological presentation is deliberate for the initial calibration.
        ordered={}
        while needed:
            ready=sorted(k for k in needed if references(nodes[k]['expression'])<=set(ordered))
            if not ready: raise ValueError('Cycle')
            for k in ready: ordered[k]=nodes[k];needed.remove(k)
        facts=facts_text(ordered)
        digest=hashlib.sha256(facts.encode()).hexdigest()
        if digest in seen: continue
        seen.add(digest)
        pair={'pair_id':f'explicit.s{seed}.{attempt:04d}','split':args.split,'seed':seed,
              'relation':r,'arithmetic':'ordinary_integer','upstream_revision':SHA,
              'source_index':attempt,'source_facts':g.vanilla_prob.strip(),
              'adaptation':'official subgraph; explicit equations; ordinary arithmetic; topological facts; no distractors',
              'facts':facts,'facts_sha256':digest,
              'nodes':ordered,'targets':{q:{'node':n,'gold':values[n],'operations':ops[n]} for q,n in zip('AB',(a,b))}}
        validate_pair(pair);pairs.append(pair);counts[r]+=1
        print(pair['pair_id'],r,facts,pair['targets'],flush=True)
        if all(v==target for v in counts.values()): break
    assert all(v==target for v in counts.values()),counts
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pairs),encoding='utf-8')
    out.with_suffix('.prompts.txt').write_text('\n\n'.join(p['pair_id']+'\n'+render(p,'AB') for p in pairs),encoding='utf-8')
    print('COMPLETE',counts)

if __name__=='__main__': main()
