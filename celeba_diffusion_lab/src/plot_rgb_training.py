"""Standalone research figure; compare validation loss at equal image exposure."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
series = []
for directory, batch, label, color in [
    ('rgb_flow_unet40k' if (root/'results/rgb_flow_unet40k/training.json').exists() else 'rgb_flow_unet5k', 32, 'U-Net 6.62M / 1 PPU', '#b36a2e'),
    ('rgb_flow_jit40k' if (root/'results/rgb_flow_jit40k/training.json').exists() else 'rgb_flow_jit20k', 32, 'JiT 6.24M / 1 PPU', '#24715b'),
    ('rgb_scaled_jit10k', 128, 'JiT 32.63M / 4 PPU', '#315bb0'),
]:
    path = root/'results'/directory/'training.json'
    if not path.exists():
        continue
    d = json.loads(path.read_text())
    rows = d['curve']
    series.append((rows, batch, label, color))
    ax.plot([r['step']*batch/1000 for r in rows],
            [r['validation_ema']['mse'] for r in rows],
            'o-', lw=2, ms=4, color=color, label=label)
for x in (160, 640, 1280):
    ax.axvline(x, color='#bbbbbb', ls=':', lw=1)
ax.set(xlabel='Training image draws (thousands; sampling with replacement)',
       ylabel='EMA validation velocity MSE (lower is better)',
       title='CelebA-64 RGB: matched exposure, fixed 512-image validation set')
ax.grid(alpha=.2)
ax.legend(frameon=False)
ax.spines[['top','right']].set_visible(False)
fig.savefig(root/'results/rgb_exposure_curves.svg')
fig.savefig(root/'results/rgb_exposure_curves.png', dpi=180)

# Keep denoising loss and generated-image quality in separate, aligned panels.
fig2, (a, b) = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
for rows, batch, label, color in series:
    a.plot([r['step']*batch/1000 for r in rows],
           [r['validation_ema']['mse'] for r in rows], lw=2, color=color, label=label)
kid_rows = {}
for directory in ('rgb_flow_evaluation', 'rgb_scaled_evaluation', 'rgb_unet_continuation_evaluation'):
    path = root/'results'/directory/'kid.json'
    if path.exists():
        kid_rows.update({r['name']: r for r in json.loads(path.read_text())['results']})
for path in (root/'results').glob('rgb_round2_evaluation_*/evaluation.json'):
    for row in json.loads(path.read_text())['rows']:
        if row['run']=='small_jit40k':
            kid_rows['flow_jit40k']=row
for label, color, pairs in [
    ('U-Net 6.62M', '#b36a2e', [('flow_unet5k',160),('flow_unet20k',640),('flow_unet40k',1280)]),
    ('JiT 6.24M', '#24715b', [('flow_jit5k',160),('flow_jit20k',640),('flow_jit40k',1280)]),
    ('JiT 32.63M', '#315bb0', [('scaled_jit1250',160),('scaled_jit5000',640),('scaled_jit10000',1280)]),
]:
    pts = [(x, kid_rows[n]) for n, x in pairs if n in kid_rows]
    b.errorbar([x for x,r in pts], [r['kid_mean']*1000 for x,r in pts],
               yerr=[r['subset_std']*1000 for x,r in pts], fmt='o-', capsize=3,
               lw=2, ms=5, label=label, color=color)
for panel in (a, b):
    panel.set_xlabel('Training image draws (thousands)')
    panel.set_xlim(0, 1320)
    panel.set_xticks([0,160,640,1280])
    for x in (160,640,1280):
        panel.axvline(x,color='#bbbbbb',ls=':',lw=.8)
    panel.grid(alpha=.15)
    panel.spines[['top','right']].set_visible(False)
a.set(title='Denoising: fixed validation set', ylabel='EMA velocity MSE (lower is better)')
b.set(title='Generation: fixed 512 RGB samples', ylabel='KID x 1,000 (lower is better)')
b.legend(frameon=False)
fig2.suptitle('CelebA-64 RGB | Same image exposure | KID error bars: subset SD, not CI', fontsize=12)
fig2.savefig(root/'results/rgb_exposure_summary.svg')
fig2.savefig(root/'results/rgb_exposure_summary.png', dpi=180)
plt.close('all')
