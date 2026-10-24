"""Shared fixed-seed RGB generation / KID protocol for the second experiment round."""
import gc
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from model import save_grid
from model_flow import FlowSampler, new_flow_model
from model_scaled import ScaledJiT
from evaluate_rgb_flow import quantize


def load_model(kind, path):
    c = torch.load(path, map_location='cpu', weights_only=False)
    m = ScaledJiT() if kind == 'scaled' else new_flow_model(kind, 3)
    m = m.cuda().eval().requires_grad_(False).to(memory_format=torch.channels_last)
    m.load_state_dict(c['ema'])
    return m, c


def score_kid(real, fake):
    rng = np.random.default_rng(42)
    n, dim, vals = 256, real.shape[1], []
    for _ in range(50):
        a = real[rng.choice(len(real), n, replace=False)].astype(np.float64)
        b = fake[rng.choice(len(fake), n, replace=False)].astype(np.float64)
        aa, bb, ab = (a@a.T/dim+1)**3, (b@b.T/dim+1)**3, (a@b.T/dim+1)**3
        vals.append(float((aa.sum()-aa.trace()+bb.sum()-bb.trace())/(n*(n-1))-2*ab.mean()))
    return {'kid_mean':float(np.mean(vals)), 'subset_std':float(np.std(vals, ddof=1)), 'values':vals}


@torch.no_grad()
def assess(m, out, name, net, real, numeric=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sampler = FlowSampler(m, batch=16, steps=50, graph=True)
    start = time.perf_counter()
    images = torch.cat([sampler.run(881+j) for j in range(32)])
    assert images.shape == (512,3,64,64) and torch.isfinite(images).all()
    row = {'name':name, 'sampling_seconds':time.perf_counter()-start,
           'outside_unit_fraction':(images.abs()>1).float().mean().item()}
    save_grid(images[:64], out/f'{name}_64.png')
    torch.save(images, out/f'{name}.pt')
    del sampler
    gc.collect();torch.cuda.empty_cache()
    if numeric:
        a = FlowSampler(m,batch=4,steps=50,graph=True,amp=True)
        amp = a.run(3341)
        del a
        gc.collect();torch.cuda.empty_cache()
        a = FlowSampler(m,batch=4,steps=50,graph=False,amp=False)
        fp32 = a.run(3341)
        row['amp_graph_vs_fp32_eager_rmse'] = (amp-fp32).square().mean().sqrt().item()
        assert torch.isfinite(fp32).all() and row['amp_graph_vs_fp32_eager_rmse'] < .02
        del a
        gc.collect();torch.cuda.empty_cache()
    x = quantize(images)
    fake = np.concatenate([net(x[i:i+32].cuda())[0].flatten(1).cpu().numpy() for i in range(0,512,32)])
    assert np.isfinite(fake).all()
    np.save(out/f'{name}_features.npy',fake)
    row.update(score_kid(real,fake))
    row['real_features_sha256'] = hashlib.sha256(real.tobytes()).hexdigest()
    return row
