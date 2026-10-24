"""Fixed, unselected RGB samples and KID for larger JiT checkpoints."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from model import save_grid, gpu_status
from model_flow import FlowSampler
from model_scaled import ScaledJiT
from evaluate_rgb_flow import quantize, write


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wait-seconds', type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    out = Path('results/rgb_scaled_evaluation')
    out.mkdir(parents=True, exist_ok=True)
    root = Path('results/rgb_scaled_jit10k')
    result = {'channels': 3, 'rows': [], 'before': gpu_status(),
              'protocol': '512 original RGB samples per checkpoint, seeds881..912, batch16, Heun50/99NFE, first64 displayed; no selection or color correction.',
              'comparison': '1250/5000 steps at global128 correspond to160k/640k image draws: small JiT5k/20k. Batch, warmup and EMA horizon differ; not a capacity-only ablation.'}
    for step in (1250, 5000, 10000):
        weights = root/f'ema_{step}.pt'
        deadline = time.monotonic() + args.wait_seconds
        while not weights.exists() or not (root/f'training_{step}.json').exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Checkpoint {step} is not ready')
            time.sleep(5)
        c = torch.load(weights, map_location='cpu', weights_only=False)
        assert c['step'] == step and c['config']['global_batch'] == 128
        m = ScaledJiT().cuda().eval().requires_grad_(False).to(memory_format=torch.channels_last)
        m.load_state_dict(c['ema'])
        sampler = FlowSampler(m, batch=16, steps=50, graph=True)
        start = time.perf_counter()
        images = torch.cat([sampler.run(881+j) for j in range(32)])
        assert images.shape == (512, 3, 64, 64) and torch.isfinite(images).all()
        name = f'scaled_jit{step}'
        torch.save(images, out/f'{name}.pt')
        save_grid(images[:64], out/f'{name}_64.png')
        row = {'name': name, 'step': step, 'training_image_draws': step*c['config']['global_batch'],
               'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
               'seconds': time.perf_counter()-start,
               'training': json.loads((root/f'training_{step}.json').read_text())}
        result['rows'].append(row)
        if step == 10000:
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
            assert torch.isfinite(fp32).all()
            result['amp_graph_vs_fp32_eager_rmse'] = (amp-fp32).square().mean().sqrt().item()
            save_grid(fp32, out/'scaled_final_fp32_4.png', columns=4, scale=3)
        del sampler, m, c
        gc.collect()
        torch.cuda.empty_cache()
        write(out/'evaluation.json', result)
        print(json.dumps({k: v for k, v in row.items() if k != 'training'}), flush=True)
    from pytorch_fid.inception import InceptionV3
    net = InceptionV3([3]).cuda().eval().requires_grad_(False)

    def features(images):
        x = quantize(images)
        assert x.shape[1] == 3
        return np.concatenate([net(x[i:i+32].cuda())[0].flatten(1).cpu().numpy()
                               for i in range(0, len(x), 32)])

    reference = Path('results/rgb_flow_evaluation/real_features.npy')
    if reference.exists():
        rf = np.load(reference)
    else:
        valid = np.load('data/valid.npy', mmap_mode='r')
        idx = np.linspace(0, len(valid)-1, 2048, dtype=np.int64)
        real = torch.from_numpy(np.array(valid[idx])).permute(0, 3, 1, 2).float()/127.5-1
        rf = features(real)
    assert rf.shape == (2048, 2048)
    np.save(out/'real_features.npy', rf)
    rows = []
    for row in result['rows']:
        name = row['name']
        fake = features(torch.load(out/f'{name}.pt', map_location='cpu', weights_only=True))
        np.save(out/f'{name}_features.npy', fake)
        rng = np.random.default_rng(42)
        vals = []
        n, dim = 256, rf.shape[1]
        for _ in range(50):
            a = rf[rng.choice(len(rf), n, replace=False)].astype(np.float64)
            b = fake[rng.choice(len(fake), n, replace=False)].astype(np.float64)
            aa, bb, ab = (a@a.T/dim+1)**3, (b@b.T/dim+1)**3, (a@b.T/dim+1)**3
            vals.append(float((aa.sum()-aa.trace()+bb.sum()-bb.trace())/(n*(n-1))-2*ab.mean()))
        rows.append({'name': name, 'kid_mean': float(np.mean(vals)),
                     'subset_std': float(np.std(vals, ddof=1)), 'values': vals})
        write(out/'kid.json', {'real_count': 2048, 'generated_per_method': 512,
              'subsets': 50, 'subset_size': 256, 'results': rows,
              'real_features_sha256': hashlib.sha256(rf.tobytes()).hexdigest(),
              'features': 'pytorch-fid0.3.0 pool3; true RGB, same quantization as small model',
              'note': 'Subset SD is not CI. One training seed, limited samples. Compare image exposure as well as model and batch.'})
        print(json.dumps({'name': name, 'kid_x1000': rows[-1]['kid_mean']*1000}), flush=True)
    result['status'] = 'complete'
    write(out/'evaluation.json', result)


if __name__ == '__main__':
    main()
