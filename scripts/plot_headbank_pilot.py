import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();root=Path(a.run)
s=json.loads((root/'analysis.json').read_text())
rr=[r for r in s['summary'] if r['phase']=='validation' and r['control'] is None and r['target']=='C']
fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
labels=[r['name'] for r in rr];counts=[r['induced'] for r in rr]
bars=ax.bar(range(len(rr)),counts,color=['#247ba0' if x=='selected' else '#94a3b8' for x in labels])
ax.set_xticks(range(len(rr)),labels,rotation=20,ha='right');ax.set_ylim(0,7);ax.set_ylabel('New final counterfactual pairs / 6');ax.set_title('Head bank heldout validation (seed 129): C interchange')
for bar,n in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,n+.1,f'{n}/6',ha='center')
fig.savefig(root/'headbank-validation.png',dpi=170)
print('Head bank validation figure saved')
