"""Standalone scientific distribution figures from saved, unmodified measurements."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

out=Path('results/gray_evaluation')
d=json.loads((out/'distribution.json').read_text())
assert len(d['rows'])==4
labels={'gray5k_eta0':'Gray training / deterministic','gray5k_eta1':'Gray training / stochastic',
        'rgb5k_eta0':'RGB training, shown gray / deterministic','rgb5k_eta1':'RGB training, shown gray / stochastic'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#b4c2b8','axes.labelcolor':'#66756e','xtick.color':'#66756e','ytick.color':'#66756e',
                     'figure.facecolor':'#fffefb','axes.facecolor':'#fffefb','svg.fonttype':'none'})

def draw(ax,values,label,color,style='-'):
    values=np.sort(values);ax.step(np.r_[0,values],np.r_[0,np.arange(1,len(values)+1)/len(values)],
                                 where='post',label=label,color=color,ls=style,lw=1.8)

def format_axis(ax,key):
    ax.set_xlim(0,255 if key=='brightness' else 128);ax.set_ylim(0,1.01)
    ax.set_xlabel('Mean image intensity [0, 255]' if key=='brightness' else 'Within-image intensity SD')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('Fraction of images')
    ax.grid(axis='y',color='#e2e6df',linewidth=.6)
    ax.set_axisbelow(True)

for row in d['rows']:
    for key in ('brightness','contrast'):
        fig,ax=plt.subplots(figsize=(4.6,2.8),layout='constrained')
        draw(ax,d['real'][key],'Real validation','#20634e')
        draw(ax,row['stats'][key],'Generated','#a96924')
        format_axis(ax,key);ax.legend(loc='lower right',frameon=False,fontsize=8)
        fig.savefig(out/(row['name']+'_'+key+'_cdf.svg'));plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
colors=['#a96924','#5674b5','#9e586f','#89924a']
for ax,key in zip(axes,('brightness','contrast')):
    draw(ax,d['real'][key],'Real validation (n=2048)','#20634e')
    for row,col in zip(d['rows'],colors):draw(ax,row['stats'][key],labels[row['name']],col,'--' if 'rgb' in row['name'] else '-')
    format_axis(ax,key)
axes[0].legend(loc='upper left',fontsize=7,frameon=False)
fig.suptitle('CelebA 64 x 64 | 5,000 training steps | 512 samples / method | DDIM 100')
fig.savefig(out/'distribution_overview.svg');fig.savefig(out/'distribution_overview.png',dpi=160);plt.close(fig)
print('Wrote 8 selectable SVGs and standalone overview SVG/PNG.')
