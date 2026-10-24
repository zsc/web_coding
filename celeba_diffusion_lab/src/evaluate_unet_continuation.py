"""Evaluate the unchanged RGB U-Net at matched 640k/1.28M image exposure."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from model import save_grid, gpu_status
from model_flow import new_flow_model, FlowSampler
from evaluate_rgb_flow import quantize, write


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wait-seconds', type=int, default=2400)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    out = Path('results/rgb_unet_continuation_evaluation')
    out.mkdir(parents=True, exist_ok=True)
    root = Path('results/rgb_flow_unet40k')
    result = {'channels': 3, 'rows': [], 'before': gpu_status(),
              'protocol': '512 original RGB samples per checkpoint, seeds881..912, batch16, Heun50/99NFE, first64 displayed; no selection or color correction.',
              'comparison': 'U-Net20k vs small JiT20k: same batch32 and640k image draws. U-Net40k vs large JiT10k:1.28M draws but different batch/capacity/EMA schedule.'}
    rf = np.load('results/rgb_flow_evaluation/real_features.npy')
    assert rf.shape == (2048, 2048)
    kid = {'real_count': 2048, 'generated_per_method': 512, 'subsets': 50, 'subset_size': 256,
           'results': [], 'real_features_sha256': hashlib.sha256(rf.tobytes()).hexdigest(),
           'features': 'pytorch-fid0.3.0 pool3; true RGB, identical quantization/reference to prior JiT/U-Net evaluation',
           'note': 'Subset SD is not CI. One training seed; limited evaluation samples.'}
    for step in (20000, 40000):
        deadline = time.monotonic() + args.wait_seconds
        while not (root/f'training_{step}.json').exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Checkpoint {step} not ready')
            time.sleep(5)
        weights = root/f'ema_{step}.pt'
        c = torch.load(weights, map_location='cpu', weights_only=False)
        assert c['step'] == step and c['config']['batch'] == 32
        assert c['config']['variant'] == 'unet' and not c['config']['grayscale']
        m = new_flow_model('unet', 3).cuda().eval().requires_grad_(False).to(memory_format=torch.channels_last)
        m.load_state_dict(c['ema'])
        sampler = FlowSampler(m, batch=16, steps=50, graph=True)
        start = time.perf_counter()
        images = torch.cat([sampler.run(881+j) for j in range(32)])
        assert images.shape == (512, 3, 64, 64) and torch.isfinite(images).all()
        name = f'flow_unet{step//1000}k'
        torch.save(images, out/f'{name}.pt')
        save_grid(images[:64], out/f'{name}_64.png')
        row = {'name': name, 'step': step, 'training_image_draws': step*32,
               'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
               'seconds': time.perf_counter()-start,
               'outside_unit_fraction': (images.abs()>1).float().mean().item(),
               'training': json.loads((root/f'training_{step}.json').read_text())}
        del sampler
        gc.collect()
        torch.cuda.empty_cache()
        sampler = FlowSampler(m, batch=4, steps=50, graph=True, amp=True)
        amp = sampler.run(3341)
        del sampler
        gc.collect()
        torch.cuda.empty_cache()
        sampler = FlowSampler(m, batch=4, steps=50, graph=False, amp=False)
        fp32 = sampler.run(3341)
        row['amp_graph_vs_fp32_eager_rmse'] = (amp-fp32).square().mean().sqrt().item()
        assert torch.isfinite(fp32).all() and row['amp_graph_vs_fp32_eager_rmse'] < .01
        save_grid(fp32, out/f'{name}_fp32_4.png', columns=4, scale=3)
        del sampler, m, c
        gc.collect()
        torch.cuda.empty_cache()
        if step == 20000:
            a = torch.load(root/f'rng_{step}.pt', map_location='cpu', weights_only=False)
            b = torch.load(f'results/rgb_flow_jit20k/rng_{step}.pt', map_location='cpu', weights_only=False)
            result['training_stream_checks_at20k'] = {
                'cuda_rng_equal': torch.equal(a['cuda_rng'], b['cuda_rng']),
                'numpy_rng_equal': a['numpy_rng'] == b['numpy_rng']}
        result['rows'].append(row)
        write(out/'evaluation.json', result)
        print(json.dumps({k: v for k, v in row.items() if k != 'training'}), flush=True)

        from pytorch_fid.inception import InceptionV3
        net = InceptionV3([3]).cuda().eval().requires_grad_(False)
        x = quantize(images)
        fake = np.concatenate([net(x[i:i+32].cuda())[0].flatten(1).cpu().numpy()
                               for i in range(0, len(x), 32)])
        assert fake.shape == (512, 2048) and np.isfinite(fake).all()
        np.save(out/f'{name}_features.npy', fake)
        del net, images, x
        gc.collect()
        torch.cuda.empty_cache()
        rng = np.random.default_rng(42)
        vals = []
        n, dim = 256, rf.shape[1]
        for _ in range(50):
            a = rf[rng.choice(len(rf), n, replace=False)].astype(np.float64)
            b = fake[rng.choice(len(fake), n, replace=False)].astype(np.float64)
            aa, bb, ab = (a@a.T/dim+1)**3, (b@b.T/dim+1)**3, (a@b.T/dim+1)**3
            vals.append(float((aa.sum()-aa.trace()+bb.sum()-bb.trace())/(n*(n-1))-2*ab.mean()))
        kid['results'].append({'name': name, 'kid_mean': float(np.mean(vals)),
                               'subset_std': float(np.std(vals, ddof=1)), 'values': vals})
        write(out/'kid.json', kid)
        print(json.dumps({'name': name, 'kid_x1000': kid['results'][-1]['kid_mean']*1000}), flush=True)
    result['status'] = 'complete'
    write(out/'evaluation.json', result)


if __name__ == '__main__':
    main()
