"""Compare backbone and training target without changing plot axes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
out=Path('results/flow_evaluation');d=json.loads((out/'distribution.json').read_text());af=json.loads(Path('results/affine_evaluation/distribution.json').read_text())
rows=d['rows']
labels=['Global U-Net / flow / 5k','JiT p4 + conv / flow / 5k','JiT p4 + conv / flow / 20k']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fffefb','axes.facecolor':'#fffefb','svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
for ax,key in zip(axes,('brightness','contrast')):
    for label,values,col in [('Real validation',d['real'][key],'#20634e')]+[(lab,r['stats'][key],col) for lab,r,col in zip(labels,rows,['#a96924','#5674b5','#9e586f'])]:
        a=np.sort(values);ax.step(np.r_[0,a],np.r_[0,np.arange(1,len(a)+1)/len(a)],where='post',label=label,color=col,lw=1.7)
    ax.set_xlim(0,255 if key=='brightness' else 128);ax.set_ylim(0,1.01);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.2)
    ax.set_xlabel('Mean image intensity [0,255]' if key=='brightness' else 'Within-image intensity SD')
axes[0].legend(loc='upper left',fontsize=7,frameon=False);fig.suptitle('512 samples / checkpoint | Heun 50 intervals / 99 NFE')
fig.savefig(out/'flow_comparison.svg');fig.savefig(out/'flow_comparison.png',dpi=160);plt.close(fig)
