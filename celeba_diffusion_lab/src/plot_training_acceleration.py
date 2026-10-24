"""Publication-style plots; timing and instrumented device costs stay separate."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1];out=root/'results/ppu_acceleration'
d=json.loads((out/'summary.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
colors={'eager':'#8c98a3','graph':'#167c80','triton':'#bc7b35','graph_triton':'#5753a3'}
names={'eager':'Baseline','graph':'Training graph','triton':'Triton RMSNorm','graph_triton':'Graph + Triton'}
fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
for ax,(kind,model) in zip(axes,d['models'].items()):
    modes=model['modes']
    for i,(mode,v) in enumerate(modes.items()):
        values=[r['median_ms'] for r in v['runs']]
        ax.bar(i,v['median_ms'],color=colors[mode],alpha=.84,width=.62)
        ax.scatter(np.full(len(values),i)+np.linspace(-.12,.12,len(values)),values,s=19,c='#243340',zorder=3)
        ax.text(i,max(values)+1,f"{v['median_ms']:.1f} ms\n{v['paired_speedup_median']:.2f}x",ha='center',va='bottom',fontsize=9)
    ax.set_xticks(range(len(modes)),[names[m] for m in modes],rotation=15,ha='right')
    ax.set_title('U-Net 6.62M' if kind=='unet' else 'JiT 32.63M')
    ax.set_ylabel('Training step wall time (ms), lower is better')
    ax.set_ylim(0,max([v['median_ms'] for v in modes.values()] or [50])*1.35)
    ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
fig.savefig(out/'training_speed.svg');fig.savefig(out/'training_speed.png',dpi=160);plt.close(fig)

profiles=d['profiles']; cats=list(dict.fromkeys(c for p in profiles.values() for c in p['groups']))
fig,ax=plt.subplots(figsize=(11,max(3,len(profiles)*.6+1.4)),layout='constrained')
palette=plt.get_cmap('tab20')
for i,(name,p) in enumerate(profiles.items()):
    left=0
    for j,c in enumerate(cats):
        ms=p['groups'].get(c,{}).get('total_ms',0)/p['steps']
        ax.barh(i,ms,left=left,color=palette(j),label=c if i==0 else None);left+=ms
ax.set_yticks(range(len(profiles)),[n.replace('large_jit','JiT').replace('unet','U-Net').replace('_',' / ') for n in profiles])
ax.invert_yaxis();ax.set_xlabel('Instrumented kernel duration sum per step (ms); not wall time')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=4,frameon=False,fontsize=8)
ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.savefig(out/'kernel_breakdown.svg');fig.savefig(out/'kernel_breakdown.png',dpi=160);plt.close(fig)
