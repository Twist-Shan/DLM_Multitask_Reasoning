"""Static scientific figures from the audited screen30 artifacts."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', type=Path, required=True)
    ap.add_argument('--analysis', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text())
    analysis = json.loads(args.analysis.read_text())
    relations = ('independent', 'shared', 'downstream')
    colors = ('#64748b', '#2563eb', '#e19027')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, family in zip(axes, ('dream', 'illada')):
        for k, mode in enumerate(('single', 'AB', 'BA')):
            values = [next(s for s in summary if s['run'] == f'screen30-{family}' and s['relation'] == rel and s['mode'] == mode) for rel in relations]
            bars = ax.bar(np.arange(3)+(k-1)*.24, [s['correct']/s['answers'] for s in values], width=.23, color=colors[k], label=mode)
            ax.bar_label(bars, labels=[f"{s['correct']}/{s['answers']}" for s in values], fontsize=7.5, padding=3)
        ax.set_title('Dream' if family == 'dream' else 'iLLaDA')
        ax.set_xticks(range(3), ['Independent', 'Shared', 'Downstream'])
        ax.set_ylim(0, 1.14); ax.set_yticks([0, .25, .5, .75, 1])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Target answer accuracy')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='lower center', ncol=3)
    fig.suptitle('10 pairs per relation, seed 123; descriptive results', fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, .96))
    fig.savefig(args.out/'igsm-screen30-accuracy.png', dpi=200)
    fig.savefig(args.out/'igsm-screen30-accuracy.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    for ax, family in zip(axes, ('dream', 'illada')):
        records = analysis['models'][family]['interventions']
        for i, r in enumerate(records):
            color = '#dc2626' if r['target_changed'] else '#2563eb'
            ax.scatter(r['delta_sum_logprob'], i, c=color, s=45, marker='x' if r['parse_error'] else 'o')
        ax.set_yticks(range(len(records)), [r['relation'][:3]+' '+r['pair_id'].split('.')[-1]+' '+r['direction'] for r in records], fontsize=8)
        ax.axvline(0, c='#aaa', linewidth=.8)
        ax.set_title('Dream' if family == 'dream' else 'iLLaDA')
        ax.set_xlabel('Change in sum of target-token\nlog marginal probabilities', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Late answer-token interventions; red: answer changed; x: parse failure', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out/'igsm-screen30-interventions.png', dpi=200)
    fig.savefig(args.out/'igsm-screen30-interventions.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
