"""Describe complete calibration runs without pooling repeated conditions as independent groups."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def analyze(run):
    meta=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if meta['status'] not in ('calibrated','calibration_below_gate'):
        raise ValueError(f'Incomplete calibration: {run}')
    rows=[json.loads(s) for s in (run/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=480 or any(r['kind']!='baseline' or r['source_kind']!='self' for r in rows):
        raise ValueError('Expected exactly 480 calibration baselines')
    index={}
    for row in rows:
        key=(row['case'],row['style'],row['relation'],row['requested'],row['unused'])
        if key in index:raise ValueError('Duplicate calibration case/condition')
        index[key]=row
    grouped=defaultdict(list)
    for row in rows:grouped[row['style']].append(row)
    styles={}
    for style,rr in grouped.items():
        failures=[r for r in rr if not r['correct']]
        styles[style]=dict(outputs=len(rr),correct=sum(r['correct'] for r in rr),
            valid=sum(r['valid'] for r in rr),
            answer_values=sum(len(r['values']) for r in rr),
            correct_answer_values=sum(r['values'][k]==v for r in rr for k,v in r['expected'].items()),
            error_counts=dict(Counter('invalid_format' if not r['valid'] else 'valid_wrong_answer' for r in failures)),
            examples=[dict(case=r['case'],relation=r['relation'],requested=r['requested'],
                unused=r['unused'],expected=r['expected'],values=r['values'],texts=r['texts'],
                prompt=r['canvas']['prompt']) for r in failures[:8]])
    cells=[]
    for style,details in meta['gate']['styles'].items():
        for key,values in details['cells'].items():
            rel,req,unused=key.split('/')
            if values['n']!=10:raise ValueError('Each cell must have ten groups')
            cells.append(dict(style=style,relation=rel,requested=req,unused=unused=='unused=True',**values))
    paired=[]
    cases=sorted({r['case'] for r in rows})
    for style in grouped:
        for relation in ('independent','shared','downstream'):
            for unused in (False,True):
                for order in ('AB','BA'):
                    for field in ('A','B'):
                        joint=[index[c,style,relation,order,unused] for c in cases]
                        separate=[index[c,style,relation,field,unused] for c in cases]
                        jc=[r['values'][field]==r['expected'][field] for r in joint]
                        sc=[r['values'][field]==r['expected'][field] for r in separate]
                        paired.append(dict(style=style,relation=relation,unused=unused,order=order,answer=field,
                            groups=len(cases),joint_correct=sum(jc),single_correct=sum(sc),
                            joint_only_correct=sum(a and not b for a,b in zip(jc,sc)),
                            single_only_correct=sum(b and not a for a,b in zip(jc,sc))))
    return dict(model=meta['model']['family'],status=meta['status'],seed=meta['seed'],
        underlying_groups=len(cases),selected_style=meta['gate']['selected_style'],
        dataset_sha256=meta['dataset_sha256'],elapsed_seconds=meta['elapsed_seconds'],
        compute=meta.get('compute'),styles=styles,cells=cells,paired_answer_accuracy=paired,
        results_sha256=hashlib.sha256((run/'results.jsonl').read_bytes()).hexdigest(),
        interpretation='Descriptive calibration only; styles and conditions reuse the same ten input groups. Error causes are not identified.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+',type=Path);p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    result=[analyze(run) for run in args.runs]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in result:
        print(r['model'],r['status'],r['selected_style'])
        for style,v in r['styles'].items():
            print(style,'correct_outputs',v['correct'],'/',v['outputs'],'valid',v['valid'],'errors',v['error_counts'])


if __name__=='__main__':main()
