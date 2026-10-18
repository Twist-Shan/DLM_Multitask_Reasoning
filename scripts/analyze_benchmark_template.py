"""Audit matched v2/v3 author-decoder runs and export a private comparison."""
import argparse
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dlm_coupling import benchmark_transfer as base, benchmark_transfer_v2 as v2, benchmark_transfer_v3 as v3


def load_run(path, version):
    data = json.loads((path / 'dataset.json').read_text(encoding='utf-8'))
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete'
    assert manifest['dataset_sha256'] == base.digest(data)
    cases = {c['id']: c for c in data['cases']}
    jobs = json.loads((path / 'jobs.json').read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in (path / 'results.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == manifest['rows'] == len(jobs) == 4
    assert len({(r['case_id'], r['requested']) for r in rows}) == 4
    prompt_module = v2 if version == 'v2' else v3
    for row, job in zip(rows, jobs):
        assert all(row[k] == value for k, value in job.items())
        case = cases[row['case_id']]
        assert row['prompt'] == prompt_module.prompt(case, row['requested'])
        expected = v2.score(case, row['generation']['text'], row['requested'])
        assert all(row[k] == value for k, value in expected.items())
        assert row['compute']['forwards'] == len(row['generation']['output_token_ids'])
    return manifest, rows, cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--previous', type=Path, required=True)
    ap.add_argument('--current', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    old_m, old, cases = load_run(args.previous, 'v2')
    new_m, new, _ = load_run(args.current, 'v3')
    for key in ('dataset_sha256', 'seed', 'model', 'upstream_sha256', 'gen_length',
                'block_length', 'steps_per_block', 'end_think_logit_boost', 'end_think_boost_power'):
        assert old_m[key] == new_m[key], key
    assert [(r['case_id'], r['requested']) for r in old] == [(r['case_id'], r['requested']) for r in new]
    escape = lambda value: html.escape(str(value))
    sections = []
    table = []
    for before, after in zip(old, new):
        for version, row in [('v2', before), ('v3', after)]:
            table.append('<tr>' + ''.join(f'<td>{escape(x)}</td>' for x in (
                row['benchmark'], row['requested'], version, row['generation']['eos_found'],
                row['format_valid'], row['scores']['B']['em'], row['scores']['B']['f1'],
                row['generation']['effective_output_tokens'], row['compute']['forwards'])) + '</tr>')
        sections.append(f'<h2>{escape(after["benchmark"])} / {escape(after["requested"])}</h2>')
        sections.append('<details><summary>Case and reference answers</summary><pre>' + escape(json.dumps(cases[after['case_id']], ensure_ascii=False, indent=2)) + '</pre></details>')
        for version, row in [('v2', before), ('v3', after)]:
            sections.append(f'<h3>{version}</h3><details><summary>Exact prompt</summary><pre>{escape(row["prompt"])}</pre></details><pre>{escape(row["generation"]["text"])}</pre>')
    body = '''<!doctype html><meta charset="utf-8"><title>Output template comparison</title>
<style>body{max-width:1080px;margin:40px auto;font:16px/1.65 system-ui;color:#18212b}td,th{border:1px solid #ccd;padding:8px}table{border-collapse:collapse}pre{white-space:pre-wrap;background:#f4f6f8;padding:16px;overflow-wrap:anywhere}</style>
<h1>Output template comparison: MuSiQue and StepGame</h1>
<p>Purpose: test whether concrete requested labels improve output validity on the same two exploratory cases. Four paired conditions, seed 123, iLLaDA, pinned author decoder, 1024-token cap. Only the output-format instructions change. Example for AB: FINAL: followed by A: [answer to question A] and B: [answer to question B]. The bracketed text must be replaced. Context and questions are unchanged.</p>
<p>EOS reports native termination. Valid means the frozen v2 parser accepts the final answer. B EM is normalized exact match; B F1 is token overlap for MuSiQue. Invalid output scores zero. Tokens count output before EOS; forwards count model calls including completion of the final block. These two cases cannot estimate benchmark accuracy or a joint-answer benefit.</p>
<table><tr><th>Benchmark</th><th>Request</th><th>Template</th><th>EOS</th><th>Valid</th><th>B EM</th><th>B F1</th><th>Tokens</th><th>Forwards</th></tr>'''
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.html').write_text(body + ''.join(table) + '</table>' + ''.join(sections), encoding='utf-8')
    args.out.with_suffix('.json').write_text(json.dumps(dict(previous=old_m,current=new_m,old=old,new=new),ensure_ascii=False,indent=2),encoding='utf-8')
    for r in new:
        print(json.dumps({k:r[k] for k in ('benchmark','requested','format_valid','scores','final_text')},ensure_ascii=False))


if __name__ == '__main__':
    main()
