"""Plot already-audited v2.4 summaries; no model execution."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('summary',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();s=json.loads(args.summary.read_text(encoding='utf-8'))
    if s['status']!='exploratory_complete':raise ValueError('Unaudited summary')
    args.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':170})
    blue,orange='#2166a5','#c6632a'
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,mode,title in zip(axes,['L31','L24_31'],['Layer 31','Sustained layers 24–31']):
        entries=[s['dependency_selectivity'][mode+'/'+side+'/all'] for side in ('all','left','right')]
        y=np.array([e['mean'] for e in entries]);lo=np.array([e['ci95'][0] for e in entries]);hi=np.array([e['ci95'][1] for e in entries])
        ax.errorbar(range(3),y,yerr=[y-lo,hi-y],fmt='o',capsize=5,color=blue)
        ax.axhline(0,color='#777777',lw=1,ls='--');ax.set_xticks(range(3),['Both sides','Parent left of B','Parent right of B'])
        ax.tick_params(axis='x',labelrotation=12);ax.set_title(title);ax.set_ylabel('Dependency-selective effect (nat / digit)')
    fig.savefig(args.out/'v24-dependency.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,event,title in zip(axes,['t0_B','final_B'],['Initial masked readout (t=0)','B committed in two forwards']):
        for filler,color in [('mask',blue),('eos',orange)]:
            values=[s['space'][f'{filler}_{k}' if k else 'mask_0'] for k in (0,2,8,32)]
            ax.plot([0,2,8,32],[100*r[event]/r['n'] for r in values],marker='o',color=color,label=filler.upper())
        ax.set_xticks([0,2,8,32]);ax.set_xlabel('Extra tail tokens');ax.set_ylabel('B exact accuracy (%)')
        ax.set_ylim(0,100);ax.set_title(title);ax.legend(frameon=False)
    fig.savefig(args.out/'v24-space-quantity.png');plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained')
    for i,(mode,label,color) in enumerate([('first','First forward only',blue),('persistent','Every B forward',orange)]):
        rows=[s['rollout'][mode+'/'+side] for side in ('left','right')]
        ax.bar(np.arange(2)+(i-.5)*.3,[100*r['after_candidate']/r['n'] for r in rows],width=.3,label=label,color=color)
    base=[s['rollout']['first/'+side] for side in ('left','right')]
    ax.scatter(range(2),[100*r['baseline_candidate']/r['n'] for r in base],marker='_',s=220,color='black',label='Clean B-first baseline',zorder=3)
    ax.set_xticks(range(2),['Parent left of B','Parent right of B']);ax.set_ylim(0,100)
    ax.set_ylabel('Committed B equals Target B ± 1 (%)');ax.set_title('Compensated Source; B commits before parent');ax.legend(frameon=False)
    fig.savefig(args.out/'v24-persistence.png');plt.close(fig)
    print(args.out)


if __name__=='__main__':main()
