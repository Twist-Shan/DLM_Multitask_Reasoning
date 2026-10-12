"""Descriptive scientific figures; no probes fitted on the tiny pilot."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();root=Path(a.run)
s=json.loads((root/'analysis.json').read_text());m=json.loads((root/'manifest.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,loc in zip(axes,('C','AB')):
    matrix=np.array([[next(r['induced'] for r in s['summary'] if r['kind']=='grid' and r['config']==f'{c}-L{l}-{loc}') for l in (8,16,24,31)] for c in ('residual','attention','mlp')])
    im=ax.imshow(matrix,vmin=0,vmax=4,cmap='Blues',aspect='auto')
    ax.set_xticks(range(4),[8,16,24,31]);ax.set_yticks(range(3),['Residual','Attention','MLP']);ax.set_xlabel('Layer (1-based)');ax.set_title('Patch positions: '+loc)
    for i in range(3):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]}/4',ha='center',va='center',color='white' if matrix[i,j]>2 else 'black')
fig.suptitle('Discovery: newly induced final counterfactual pairs (seed 127)')
fig.savefig(root/'component-grid.png',dpi=170);plt.close(fig)
hh=sorted([r for r in s['summary'] if r['kind']=='head_discovery'],key=lambda r:r['head'])
fig,ax=plt.subplots(figsize=(10,3),constrained_layout=True)
ax.bar([r['head'] for r in hh],[r['induced'] for r in hh]);ax.set_ylim(0,4.5);ax.set_xticks(range(0,32,2));ax.set_xlabel('Query head index (0-based)');ax.set_ylabel('New final pairs / 4');ax.set_title('Single-head output interchange: '+s['attention_chosen'])
fig.savefig(root/'head-scan.png',dpi=170);plt.close(fig)
# Normalized Euclidean changes, averaged over numeric positions of each variable.
z=np.load(root/'representations.npz');assert len(z.files)==60
rep=[]
for split in ('discovery','validation'):
    for stage in ('initial','mid'):
        for variant in ('C','same'):
            distances=[]
            for c in m['data']:
                if c['split']!=split:continue
                base=z[c['id']+'_base_'+stage].astype(float);donor=z[c['id']+'_'+variant+'_'+stage].astype(float)
                assert base.shape==(4,10,4096)
                d=np.linalg.norm(donor-base,axis=-1)/np.maximum(np.linalg.norm(base,axis=-1),1e-8)
                distances.append(d.reshape(4,5,2).mean(-1))
            means=np.mean(distances,axis=0)
            rep.append(dict(split=split,stage=stage,variant=variant,relative_l2_by_layer_variable=means.tolist()))
(root/'representation-distances.json').write_text(json.dumps(rep,indent=2))
fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
records=[r for r in rep if r['split']=='validation'];vmax=max(np.max(r['relative_l2_by_layer_variable']) for r in records)
for ax,r in zip(axes.flat,records):
    im=ax.imshow(r['relative_l2_by_layer_variable'],vmin=0,vmax=vmax,cmap='viridis',aspect='auto');ax.set_xticks(range(5),list('CUVAB'));ax.set_yticks(range(4),[8,16,24,31]);ax.set_ylabel('Layer');ax.set_title(r['stage']+' / '+('C + 1' if r['variant']=='C' else 'Same C, different inputs'))
fig.colorbar(im,ax=axes,shrink=.7,label='Relative L2 change (descriptive only)');fig.suptitle('Residual changes: 6 validation cases; before / after generated intermediates')
fig.savefig(root/'representation-distances.png',dpi=170);plt.close(fig)
print('Three figures and representation distance summaries saved')
