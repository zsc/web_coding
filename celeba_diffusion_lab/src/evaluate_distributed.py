"""Shard the existing 32 fixed generation seeds over all 16 reserved PPUs."""
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from eval_protocol import load_model, score_kid
from evaluate_rgb_flow import quantize
from model import save_grid
from model_flow import FlowSampler, flow_loss


@torch.no_grad()
def evaluate(root, step, kind):
    root = Path(root)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 16
    out = root/'evaluation'
    if rank == 0:
        out.mkdir(exist_ok=True)
    dist.barrier()
    from pytorch_fid.inception import InceptionV3
    net = InceptionV3([3]).cuda().eval().requires_grad_(False)
    result_path = out/'evaluation.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {'rows': [],
        'protocol': 'RGB64;512 fixed samples seeds881..912 in original order;Heun50/99NFE;2048 real;KID50 subsets256 seed42;subsetSD not CI;16-rank sharded sampling, original model implementation.'}
    for prefix in ('ema', 'slow_ema'):
        start = time.perf_counter()
        weights = root/f'{prefix}_{step}.pt'
        m, c = load_model(kind, weights)
        sampler = FlowSampler(m, batch=16, steps=50, graph=True)
        images = torch.cat([sampler.run(881+rank*2+j) for j in range(2)])
        assert images.shape == (32, 3, 64, 64) and torch.isfinite(images).all()
        del sampler
        gc.collect()
        torch.cuda.empty_cache()
        fake = net(quantize(images).cuda())[0].flatten(1).contiguous()
        assert torch.isfinite(fake).all()
        image_parts = [torch.empty_like(images, device='cuda') for _ in range(world)]
        feature_parts = [torch.empty_like(fake) for _ in range(world)]
        dist.all_gather(image_parts, images.cuda())
        dist.all_gather(feature_parts, fake)
        if rank == 0:
            images = torch.cat(image_parts).cpu()
            features = torch.cat(feature_parts).cpu().numpy()
            name = f'{root.name}_{step}_{prefix}'
            row = {'name': name, 'step': step, 'weights_prefix': prefix,
                   'training_image_draws': c['training_image_draws'], 'batch': c['config']['global_batch'],
                   'sampling_and_features_seconds': time.perf_counter()-start,
                   'outside_unit_fraction': (images.abs()>1).float().mean().item(),
                   'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest()}
            save_grid(images[:64], out/f'{name}_64.png')
            torch.save(images, out/f'{name}.pt')
            np.save(out/f'{name}_features.npy', features)
            real = np.load('results/rgb_flow_evaluation/real_features.npy')
            row.update(score_kid(real, features))
            row['real_features_sha256'] = hashlib.sha256(real.tobytes()).hexdigest()
            amp = FlowSampler(m, batch=4, steps=50, graph=True, amp=True)
            ai = amp.run(3341)
            del amp
            gc.collect()
            torch.cuda.empty_cache()
            fp32 = FlowSampler(m, batch=4, steps=50, graph=False, amp=False)
            fi = fp32.run(3341)
            row['amp_graph_vs_fp32_eager_rmse'] = (ai-fi).square().mean().sqrt().item()
            assert torch.isfinite(fi).all() and row['amp_graph_vs_fp32_eager_rmse'] < .02
            del fp32
            # Fixed512 validation matches the preceding single-PPU experiment.
            valid = np.load('data/valid.npy', mmap_mode='r')
            vi = np.linspace(0, len(valid)-1, 512, dtype=np.int64)
            v = torch.from_numpy(np.array(valid[vi])).cuda().permute(0,3,1,2).float()/127.5-1
            vg = torch.Generator(device='cuda').manual_seed(2401)
            vt = torch.sigmoid(torch.randn(512, device='cuda', generator=vg)*.8-.8)
            vn = torch.randn(v.shape, device='cuda', generator=vg)
            a = vt[:,None,None,None]
            vx = (a*v+(1-a)*vn).to(memory_format=torch.channels_last)
            row['validation_mse'] = float(np.mean([flow_loss(m(vx[j:j+32],vt[j:j+32]),v[j:j+32],vt[j:j+32]).item() for j in range(0,512,32)]))
            # Check seed order / rank sharding against the original sampler.
            if not result['rows'] and prefix == 'ema':
                original = FlowSampler(m, batch=16, steps=50, graph=True)
                serial = torch.cat([original.run(881+j) for j in (0,1,30,31)])
                distributed = torch.cat((images[:32], images[-32:]))
                row['distributed_vs_serial_probe_max_abs'] = float((serial-distributed).abs().max())
                assert row['distributed_vs_serial_probe_max_abs'] < 2e-3
                del original
            result['rows'].append(row)
            result.update(last_step=step, status='complete')
            tmp = result_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(result, indent=2))
            tmp.replace(result_path)
            print(json.dumps({'event':'quality', **{k:v for k,v in row.items() if k!='values'}}), flush=True)
        del m, c, image_parts, feature_parts, images, fake
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    del net
    gc.collect()
    torch.cuda.empty_cache()

