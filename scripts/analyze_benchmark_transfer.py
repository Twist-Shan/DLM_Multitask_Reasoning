"""Question-level counts and paired differences; no small-sample significance claims."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dlm_coupling import benchmark_transfer as task


def analyze(rows):
    groups = defaultdict(list)
    paired = defaultdict(dict)
    for r in rows:
        key = (r['benchmark'], r['context'], r['requested'])
        groups[key].append(r)
        pair_key = (r['benchmark'], r['context'], r['case_id'])
        if r['requested'] in paired[pair_key]:
            raise ValueError('Duplicate case/condition')
        paired[pair_key][r['requested']] = r
    summary = []
    for key, rs in sorted(groups.items()):
        summary.append(dict(benchmark=key[0], context=key[1], requested=key[2], n=len(rs),
                            format_valid=sum(r['format_valid'] for r in rs),
                            B_correct=sum(r['scores']['B']['em'] for r in rs),
                            B_mean_f1=sum(r['scores']['B']['f1'] for r in rs)/len(rs)))
    deltas = defaultdict(list)
    for (benchmark, context, _), conditions in paired.items():
        if 'B' not in conditions:
            continue
        for condition in ('AB', 'BA', 'RB'):
            if condition in conditions:
                deltas[(benchmark, context, condition)].append(
                    conditions[condition]['scores']['B']['em'] - conditions['B']['scores']['B']['em'])
    return dict(summary=summary, paired_differences=[dict(benchmark=b, context=c, comparison=r+'-B',
                    n=len(values), delta_em=sum(values)/len(values), improved=values.count(1),
                    worsened=values.count(-1)) for (b,c,r), values in sorted(deltas.items())])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in (args.run/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    data = json.loads((args.run/'dataset.json').read_text(encoding='utf-8'))
    manifest = json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
    by_id = {c['id']: c for c in data['cases']}
    for row in rows:
        rescored = task.score(by_id[row['case_id']], row['generation']['text'], row['requested'])
        if any(row[key] != rescored[key] for key in rescored):
            raise ValueError('Stored score does not match regenerated score')
    result = analyze(rows)
    result['manifest'] = manifest
    result['by_hops'] = {str(h): analyze([r for r in rows if r['hops'] == h]) for h in (2,3,4)}
    selected = {r['case_id'] for r in rows}
    spatial = [by_id[k] for k in selected if by_id[k]['benchmark'] == 'stepgame']
    result['stepgame_same_A_B'] = dict(n=len(spatial), same=sum(c['answers']['A']==c['answers']['B'] for c in spatial))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    esc = html.escape
    def table(data):
        if not data:
            return '<p>尚无结果。</p>'
        keys = list(data[0])
        return '<table><tr>'+''.join('<th>'+esc(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(str(r[k]))+'</td>' for k in keys)+'</tr>' for r in data)+'</table>'
    body = '''<!doctype html><meta charset="utf-8"><title>Benchmark transfer v1</title>
    <style>body{max-width:1100px;margin:40px auto;font:16px/1.7 system-ui;color:#172633;padding:0 24px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd4dc;padding:8px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:16px}summary{cursor:pointer}h2{margin-top:38px}</style>
    <h1>自然文本与空间推理：首次迁移试验</h1>
    <p>目的：检验联合回答相关子问题是否改变原始问题 B 的正确率，并与顺序变化和无关子问题比较。
    本轮为 iLLaDA 探索性自由生成校准，尚未进行 Source/Target activation patching。</p>
    <h2>设置与评分</h2><p>预定设置：seed=123；MuSiQue 与 StepGame 各按 2/3/4 跳取 3 题，共18题。实际执行数见下表。B、AB、BA、RB 使用相同上下文与128 token输出预算，每次forward提交4个token。
    A为上游子问题；R为另一问题或断开空间实体对。MuSiQue full为原始全部段落加R段落，support为标准支持段落加R段落，属于提供标准检索结果的诊断条件。
    StepGame 按符号解析后的实际查询路径长度分组，保留原始k字段。</p>
    <p>format_valid要求按顺序输出指定的 LABEL: answer 行。B_correct为B完全正确数，分母n为实际运行题数，格式错误记0。
    MuSiQue的EM和token F1按标准规范化、取答案别名中的最高分；StepGame为方向标签完全匹配。
    delta_em是同一题联合条件减B-only的平均差，统计单位是题；improved/worsened为改善/下降题数。这里不报告小样本显著性。</p>
    <p>当前结论：设置已迁移到原始自然文本问题，但增加了R段落和子问题，因此分数不能视为官方benchmark成绩。</p>
    <h2>执行状态</h2><pre>'''+esc(json.dumps(manifest,ensure_ascii=False,indent=2))+'''</pre>
    <h2>结果</h2>'''+table(result['summary'])+'<h2>配对变化</h2>'+table(result['paired_differences'])+'''
    <p>当前结论：这些行为差异用于筛选下一轮适合研究的任务。它们不能证明A区域中的未提交状态被B使用，也不能证明DLM优于ARLM。
    格式失败、标准支持条件和完整上下文条件应分别检查；每个深度仅3题，结论待扩大样本验证。</p>
    <h2>原始输出与完整 prompt</h2><p>以下记录保留失败样例。每个折叠项对应一次实际生成。</p>'''
    body+='<p>StepGame中A、B答案恰好相同的题数：'+str(result['stepgame_same_A_B']['same'])+'/'+str(len(spatial))+'。不能以相同答案的改善单独排除复制解释。</p>'
    for r in rows:
        title=f"{r['benchmark']} {r['case_id']} {r['context']} {r['requested']} | B EM={r['scores']['B']['em']}"
        body+='<details><summary>'+esc(title)+'</summary><h3>Prompt</h3><pre>'+esc(r['prompt'])+'</pre><h3>Output</h3><pre>'+esc(r['generation']['text'])+'</pre></details>'
    body+='''<h2>出处与解释边界</h2><p><a href="https://aclanthology.org/2022.tacl-1.31/">MuSiQue原论文</a>采用有依赖的单跳问题组合，并保留支持段落与分解；
    <a href="https://github.com/ShiZhengyan/StepGame">StepGame作者仓库</a>为合成空间任务；
    <a href="https://arxiv.org/abs/2401.03991">StepGame模板审计</a>说明应先核对语言模板和标签。
    本地docs/benchmark_transfer.md记录了适配限制与下一步因果实验要求。</p>'''
    args.out.write_text(body, encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
