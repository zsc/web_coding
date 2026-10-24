"""Saved measurements only; compare identical sampler settings across architectures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
out=Path('results/affine_evaluation');d=json.loads((out/'distribution.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.facecolor':'#fffefb','axes.facecolor':'#fffefb','svg.fonttype':'none'})
labels={'gray5k':'A / additive time','time_affine5k':'B / time affine','global_affine5k':'C / global affine'}
colors=['#a96924','#5674b5','#9e586f']
def draw(ax,values,label,color):
    a=np.sort(values);ax.step(np.r_[0,a],np.r_[0,np.arange(1,len(a)+1)/len(a)],where='post',label=label,color=color,lw=1.8)
for eta in (0,1):
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for ax,key in zip(axes,('brightness','contrast')):
        draw(ax,d['real'][key],'Real validation','#20634e')
        for label,color in zip(labels,colors):
            r=next(r for r in d['rows'] if r['name']==f'{label}_eta{eta}')
            draw(ax,r['stats'][key],labels[label],color)
        ax.set_xlim(0,255 if key=='brightness' else 128);ax.set_ylim(0,1.01)
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.2)
        ax.set_xlabel('Mean image intensity [0, 255]' if key=='brightness' else 'Within-image intensity SD')
    axes[0].legend(loc='upper left',fontsize=8,frameon=False)
    fig.suptitle(f'Grayscale CelebA | 5,000 updates | 512 samples / model | DDIM 100, eta={eta}')
    fig.savefig(out/f'affine_eta{eta}_cdf.svg');fig.savefig(out/f'affine_eta{eta}_cdf.png',dpi=160);plt.close(fig)
print('Wrote affine CDF comparisons.')
