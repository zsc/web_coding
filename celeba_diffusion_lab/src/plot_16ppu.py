"""Completed16PPU experiment curves; fixed evaluation points, no smoothing."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
base=root/'results/rgb_16ppu'
runs=json.loads((base/'completion_summary.json').read_text())['runs']
colors={'unet_lr5e5':'#237b61','unet_lr2e4':'#213e59','jit_lr5e5':'#bb7232','jit_lr2e4':'#855a9b'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
for r in runs:
    rows=[x for x in r['rows'] if x['weights_prefix']=='slow_ema']
    x=[v['training_image_draws']/1e6 for v in rows]
    label=('U-Net' if r['run'].startswith('unet') else 'JiT')+' / '+('5e-5' if '5e5' in r['run'] else '2e-4')
    style={'color':colors[r['run']],'marker':'o','label':label}
    axes[0].plot(x,[v['validation_mse'] for v in rows],**style)
    axes[1].errorbar(x,[v['kid_mean']*1000 for v in rows],yerr=[v['subset_std']*1000 for v in rows],capsize=3,**style)
for ax in axes:
    ax.set_xlabel('Cumulative training image draws (millions)')
    ax.set_xticks([2.56,3.84,5.12])
    ax.grid(alpha=.17)
axes[0].set_title('Fixed validation velocity loss / slow EMA')
axes[0].set_ylabel('MSE (lower is better)')
axes[1].set_title('Generation quality / slow EMA')
axes[1].set_ylabel('KID x1,000 (lower is better)')
axes[1].legend(fontsize=9,loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
fig.savefig(base/'final_curves.svg')
fig.savefig(base/'final_curves.png',dpi=150)

