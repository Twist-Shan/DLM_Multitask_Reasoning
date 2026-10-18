"""Audit and summarize the expanded concrete-label author-decoder pilot."""
import argparse
import html
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dlm_coupling import benchmark_transfer as base, benchmark_transfer_v2 as scoring, benchmark_transfer_v3 as prompt_task
from run_benchmark_official_diagnostic import make_jobs
from analyze_benchmark_transfer_v2 import summarize


def audit(run):
    read=lambda name: json.loads((run/name).read_text(encoding='utf-8'))
    manifest,data,jobs=read('manifest.json'),read('dataset.json'),read('jobs.json')
    assert manifest['prompt_version']=='v3' and manifest['scope']=='expanded'
    assert manifest['dataset_sha256']==base.digest(data)
    cases,expected=make_jobs(data,prompt_task,'expanded')
    assert jobs==expected and len(jobs)==manifest['planned_rows']==72
    by_id={c['id']:c for c in cases}
    rows=[json.loads(s) for s in (run/'results.jsonl').read_text(encoding='utf-8').splitlines() if s]
    assert len(rows)==manifest['rows']<=len(jobs)
    assert len({(r['case_id'],r['requested']) for r in rows})==len(rows)
    for row,job in zip(rows,jobs):
        assert all(row[k]==v for k,v in job.items())
        case=by_id[row['case_id']]
        score=scoring.score(case,row['generation']['text'],row['requested'])
        assert all(row[k]==v for k,v in score.items())
        assert not row['generation']['gold_used_for_generation']
        assert row['compute']['forwards']==len(row['generation']['output_token_ids'])
        row.update(sampler='author32_boost100',hops=case['hops'])
    if manifest['status']=='complete':
        assert len(rows)==72
        assert all(sum(r['compute'][k] for r in rows)==v for k,v in manifest['compute'].items())
    result=summarize(rows)
    result.update(manifest=manifest,completed=len(rows),planned=72,
                  by_hops={str(h):summarize([r for r in rows if r['hops']==h]) for h in (2,3,4)})
    different={c['id'] for c in cases if c['benchmark']=='stepgame' and c['answers']['A']!=c['answers']['B']}
    result['spatial_different_answers']=summarize([r for r in rows if r['case_id'] in different])
    # Both-valid comparison is descriptive: conditioning on validity can select cases.
    pairs={(r['case_id'],r['requested']):r for r in rows}
    result['both_valid_pairs']=[]
    for benchmark in ('musique','stepgame'):
        for request in ('AB','BA','RB'):
            selected=[]
            for c in cases:
                a,b=pairs.get((c['id'],'B')),pairs.get((c['id'],request))
                if c['benchmark']==benchmark and a and b and a['format_valid'] and b['format_valid']:
                    selected.append(b['scores']['B']['em']-a['scores']['B']['em'])
            result['both_valid_pairs'].append(dict(benchmark=benchmark,comparison=request+'-B',n=len(selected),
                improved=selected.count(1),worsened=selected.count(-1),
                delta_em=sum(selected)/len(selected) if selected else None))
    return result,rows,by_id


def table(headers,records):
    esc=lambda x:html.escape(str(x))
    return '<table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in records)+'</table>'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    result,rows,cases=audit(args.run)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>扩展 benchmark 生成校准</title>',
      '<style>body{font:17px/1.7 system-ui;max-width:1120px;margin:40px auto;padding:20px;color:#24303d}table{border-collapse:collapse;width:100%}th,td{padding:8px;border:1px solid #ccd}pre{white-space:pre-wrap;background:#f4f6f8;padding:16px;overflow-wrap:anywhere}h2{margin-top:36px}</style>',
      '<h1>MuSiQue 与 StepGame：扩大具体标签模板实验</h1>',
      f'<p>状态：{html.escape(result["manifest"]["status"])}，已完成 {len(rows)}/72 条。每套任务 9 道题，2/3/4 跳各 3 道，每题 B-only、AB、BA、RB。seed 123。此前四条诊断不计入本次样本量。</p>',
      '<h2>目的与设置</h2><p>检验具体输出标签在更大题组上的可靠性，以及联合请求对同题 B 答案的影响。A 是上游问题，R 是选取的无关问题，AB/BA/RB 表示请求顺序。各条件背景相同。iLLaDA 固定版本、BF16、A100，作者解码每块 32 tokens、32 次前向，最大 1024 tokens，结束推理 boost 100、指数 3；v3 模板与 v2 评分规则固定。这是既有探索题组的扩展，不能视作独立确认集。</p>',
      '<p>说明性例子：X 在 Y 右侧，Y 在 Z 上方。A 问 X 相对 Y，B 问 X 相对 Z，答案为 right、upper-right。B-only 只请求 B；AB 同时请求上游 A 与 B。模型可自由输出推理，最终以 FINAL: 和具体标签结束。此例仅解释设计。</p>',
      '<h2>主指标</h2><p>有效格式数要求最终标签完整、顺序正确、推理标签闭合。B EM 为格式通过且规范化答案完全匹配；无效格式计零。MuSiQue F1 按参考别名取最高词元重叠分数，F1=2×重叠词元数/(预测词元数+参考词元数)。EOS 表示预算内生成结束符。表中分母为生成条件数；同一道题的四个条件不视为四道独立题。</p>']
    text.append(table(['任务','请求','n','EOS','有效格式','B EM 正确数','B 平均 F1','前向次数'],[
        [r[k] for k in ('benchmark','request','n','eos','valid','B_correct','B_f1','forwards')] for r in result['summary']]))
    text+=['<h2>同题配对变化</h2><p>ΔEM=平均(联合请求 B EM−B-only EM)，正值表示联合请求较好。“改善/变差”计同题从错到对/从对到错的数量。格式错误也会影响该指标，因此另列两种请求都合规的子集；子集比较有选择偏差，只作描述。</p>']
    for title,source in [('全部配对',result['paired']),('两种请求都合规',result['both_valid_pairs'])]:
        text.append('<h3>'+title+'</h3>'+table(['任务','比较','n','ΔEM','改善','变差'],[[r[k] for k in ('benchmark','comparison','n','delta_em','improved','worsened')] for r in source]))
    text+=['<h2>分层与限制</h2><p>每个跳数仅三道题，所有结果为探索性行为证据。正确答案前缀、生成顺序和自由推理同时变化，不能由行为差值直接推断 masked 区域的信息交换；尚未进行这些 benchmark 上的 patching。</p>']
    for hops,value in result['by_hops'].items():
        text.append('<h3>'+hops+' 跳</h3>'+table(['任务','请求','n','有效格式','B EM 正确数'],[[r[k] for k in ('benchmark','request','n','valid','B_correct')] for r in value['summary']]))
    text.append('<details><summary>StepGame 排除 A=B 答案题后的统计</summary><pre>'+html.escape(json.dumps(result['spatial_different_answers'],ensure_ascii=False,indent=2))+'</pre></details>')
    text.append('<h2>逐题精确 prompt、参考答案与原始生成</h2>')
    for r in rows:
        text.append('<details><summary>'+html.escape(f'{r["benchmark"]} | {r["case_id"]} | {r["hops"]} hop | {r["requested"]} | valid={r["format_valid"]} | B EM={r["scores"]["B"]["em"]}')+'</summary><h3>参考答案</h3><pre>'+html.escape(json.dumps(cases[r['case_id']]['answers'],ensure_ascii=False))+'</pre><h3>Prompt</h3><pre>'+html.escape(r['prompt'])+'</pre><h3>原始生成</h3><pre>'+html.escape(r['generation']['text'])+'</pre></details>')
    text.append('<details><summary>运行元数据</summary><pre>'+html.escape(json.dumps(result['manifest'],ensure_ascii=False,indent=2))+'</pre></details></html>')
    args.out.with_suffix('.html').write_text(''.join(text),encoding='utf-8')
    print(json.dumps(dict(completed=len(rows),summary=result['summary'],paired=result['paired']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
