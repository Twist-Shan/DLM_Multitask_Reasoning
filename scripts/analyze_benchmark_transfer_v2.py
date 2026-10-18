"""Audit frozen v2 scores and summarize paired, exploratory benchmark results."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dlm_coupling import benchmark_transfer as base, benchmark_transfer_v2 as task


def summarize(rows):
    groups = defaultdict(list)
    paired = defaultdict(dict)
    for row in rows:
        groups[(row['benchmark'], row['sampler'], row['requested'])].append(row)
        key = (row['benchmark'], row['sampler'], row['case_id'])
        if row['requested'] in paired[key]:
            raise ValueError('Duplicate case/sampler/request')
        paired[key][row['requested']] = row
    summary = []
    for (benchmark, sampler, request), rs in sorted(groups.items()):
        valid = [r for r in rs if r['format_valid']]
        summary.append(dict(benchmark=benchmark, sampler=sampler, request=request,
            n=len(rs), valid=len(valid), eos=sum(r['generation']['eos_found'] for r in rs),
            B_correct=sum(r['scores']['B']['em'] for r in rs),
            B_f1=sum(r['scores']['B']['f1'] for r in rs)/len(rs),
            B_em_given_valid=sum(r['scores']['B']['em'] for r in valid)/len(valid) if valid else None,
            forwards=sum(r['compute']['forwards'] for r in rs),
            token_positions=sum(r['compute']['processed_token_positions'] for r in rs)))
    differences = defaultdict(list)
    for (benchmark, sampler, _), conditions in paired.items():
        if 'B' not in conditions:
            continue
        for request in ('AB', 'BA', 'RB'):
            if request in conditions:
                differences[(benchmark, sampler, request)].append(
                    conditions[request]['scores']['B']['em']-conditions['B']['scores']['B']['em'])
    return dict(summary=summary, paired=[dict(benchmark=b, sampler=s, comparison=r+'-B',
        n=len(v), delta_em=sum(v)/len(v), improved=v.count(1), worsened=v.count(-1))
        for (b,s,r), v in sorted(differences.items())])


def audit(run):
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    data = json.loads((run/'dataset.json').read_text(encoding='utf-8'))
    jobs = json.loads((run/'jobs.json').read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in (run/'results.jsonl').read_text(encoding='utf-8').splitlines() if s]
    assert manifest['protocol'] == task.PROTOCOL
    assert manifest['dataset_sha256'] == base.digest(data)
    assert manifest['rows'] == len(rows)
    by_id = {c['id']: c for c in data['cases']}
    key = lambda r: (r['case_id'], r['sampler'], r['requested'])
    expected = {key(j): j for j in jobs if j.get('eligible', True)}
    assert len(expected) == sum(j.get('eligible', True) for j in jobs)
    for r in rows:
        assert key(r) in expected
        assert r['prompt'] == expected[key(r)]['prompt'] == task.prompt(by_id[r['case_id']], r['requested'])
        scored = task.score(by_id[r['case_id']], r['generation']['text'], r['requested'])
        assert all(r[k] == v for k,v in scored.items()), 'Score mismatch'
        g = r['generation']
        assert not g['gold_used_for_generation']
        assert g['nfe'] == len(g['steps']) == r['compute']['forwards']
        writes = [w for step in g['steps'] for w in step['writes']]
        assert len({w['output_position'] for w in writes}) == len(writes) == len(g['output_token_ids'])
        assert all(g['output_token_ids'][w['output_position']] == w['token_id'] for w in writes)
    result = summarize(rows)
    if manifest['status'] == 'complete':
        assert {key(r) for r in rows} == set(expected)
        assert all(sum(r['compute'][k] for r in rows) == v for k,v in manifest['compute'].items())
    result.update(manifest=manifest, eligible_jobs=len(expected), completed_rows=len(rows),
        by_hops={str(h): summarize([r for r in rows if r['hops']==h]) for h in (2,3,4)})
    return result, rows, by_id


def table(records):
    if not records:
        return '<p>No observations.</p>'
    esc = lambda x: html.escape(str(round(x,4) if isinstance(x,float) else x))
    keys = list(records[0])
    return '<table><tr>'+''.join('<th>'+esc(k)+'</th>' for k in keys)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+esc(r[k])+'</td>' for k in keys)+'</tr>' for r in records)+'</table>'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result, rows, by_id = audit(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    esc = html.escape
    parts = ['<!doctype html><meta charset="utf-8"><title>MuSiQue & StepGame v2</title>',
        '<style>body{font:16px/1.65 system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#202c38}table{border-collapse:collapse;font-size:14px}th,td{border:1px solid #ccd5de;padding:7px}pre{white-space:pre-wrap;background:#f1f4f7;padding:16px}details{margin:16px 0}h2{margin-top:36px}</style>',
        '<h1>MuSiQue 与 StepGame：v2 生成校准</h1>',
        '<p>目的：先检验模型能否稳定完成推理与答案，再探索联合请求是否改善目标问题 B。A 是相关上游问题，R 是选取的无关问题；AB/BA 表示请求的答案顺序，不保证实际推理或 token 提交顺序。</p>',
        '<p>设置：seed 123，iLLaDA，原生 chat template，完整上下文。主实验按 32-token 块逐步追加，块内每次提交四个最高置信度 token，最大输出 1024 tokens。四个完整画布条件只作同题诊断；两种方法并不匹配总计算量，也同时改变了可见 mask 空间与提交限制。</p>',
        '<p>评分在运行前固定：只读取完成的 think 区域之后、FINAL: 标记之后的标签答案；也接受直接标签答案。未完成推理、错误标签或顺序均判格式失败，不从推理里挑正确答案。EM 为规范化后完全匹配；MuSiQue F1 为答案词元重叠 F1。格式失败计入总分母，同时报告有效格式条件下的 EM。</p>',
        '<p>来源：<a href="https://github.com/StonyBrookNLP/musique">MuSiQue</a>（文档多跳问答）；<a href="https://github.com/ShiZhengyan/StepGame">StepGame</a>（合成空间关系）。这是适配后的探索子集，并非官方排行榜成绩。MuSiQue 使用公开镜像；StepGame 按语义解析与标签一致性筛选，存在筛选偏差。</p>',
        '<h2>总体结果</h2>',table(result['summary']),
        '<p>n 是生成条件数；valid 是最终格式有效数；eos 是生成终止符出现数；B_correct 为 B 的 EM 正确数。B_f1 包含格式失败；B_em_given_valid 只作诊断。forwards 为前向次数，token_positions 是每次前向输入长度之和，并非 FLOPs。</p>',
        '<h2>同题配对比较</h2>',table(result['paired']),
        '<p>delta_em 是同题联合请求减 B-only 的平均 EM 差；improved/worsened 分别为错转对、对转错的题数。主实验每个 benchmark 最多九题，完整画布比较每个 benchmark 仅一题；结果仅描述观察，不足以确认普遍联合收益，更不能作为 activation 信息共享的因果证据。</p>',
        '<h2>逐题 prompt、答案与原始生成</h2>']
    for r in rows:
        parts.append('<details><summary>'+esc(f"{r['benchmark']} | {r['case_id']} | {r['sampler']} | {r['requested']} | valid={r['format_valid']} | B EM={r['scores']['B']['em']}")+'</summary><h3>Prompt</h3><pre>'+esc(r['prompt'])+'</pre><h3>参考答案</h3><pre>'+esc(json.dumps(by_id[r['case_id']]['answers'],ensure_ascii=False,indent=2))+'</pre><h3>原始生成</h3><pre>'+esc(r['generation']['text'])+'</pre><h3>评分</h3><pre>'+esc(json.dumps({k:r[k] for k in ('final_text','extraction_route','scores')},ensure_ascii=False,indent=2))+'</pre></details>')
    parts.extend(['<h2>运行记录</h2><pre>',esc(json.dumps(result['manifest'],ensure_ascii=False,indent=2)),'</pre>'])
    args.out.with_suffix('.html').write_text('\n'.join(parts),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('completed_rows','summary','paired')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
