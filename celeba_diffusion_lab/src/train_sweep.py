"""RGB JiT batch/LR sweep with full-state continuation from scaled training."""
import argparse
import copy
import datetime
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import save_grid, gpu_status
from model_flow import FlowSampler, flow_loss
from model_scaled import ScaledJiT


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--global-batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--out', required=True)
    p.add_argument('--resume-from', default=None, help='Saved checkpoint directory with state.pt and per-rank RNG')
    args = p.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    assert args.global_batch % world == 0
    batch = args.global_batch // world
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    if world > 1:
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10))
        probe = torch.tensor(float(rank + 1), device='cuda')
        dist.all_reduce(probe)
        assert probe.item() == world * (world + 1) / 2

    def barrier():
        if world > 1:
            dist.barrier()

    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        assert not (out/'training.json').exists(), 'Use a fresh output directory'
    barrier()
    torch.manual_seed(20261024)
    model = ScaledJiT().cuda().to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False,
                  gradient_as_bucket_view=True) if world > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, fused=True)
    scaler = torch.amp.GradScaler('cuda', init_scale=4096.)
    params, ep = list(model.parameters()), list(ema.parameters())
    data = np.load('data/train.npy', mmap_mode='r')
    assert data.shape == (162770, 64, 64, 3)
    rng = np.random.default_rng(20261024 + rank)
    torch.manual_seed(20261024 + 1000 + rank)
    host = torch.empty(batch, 64, 64, 3, dtype=torch.uint8, pin_memory=True)
    md = {'config': vars(args), 'world_size': world, 'local_batch': batch,
          'parameters': sum(v.numel() for v in params), 'model': model.experiment_config,
          'channels': 3, 'torch': torch.__version__, 'curve': [],
          'recipe': 'Same RGB cache, patch4/conv4x64, direct-x/velocity loss, t=sigmoid(N(-.8,.8^2)), AdamW with configured LR, wd1e-4, clip1, FP16 autocast.',
          'changes': 'Fixed32.63M RGB JiT. Sweep batch128/512 and LR2e-4/8e-4. Warmup16000 image draws and EMA horizon scaled by image count; same evaluation protocol.',
          'sampling': 'Independent rank-specific random streams, uniform training draws with replacement and horizontal flip.',
          'before': gpu_status() if rank == 0 else None}
    if rank == 0:
        valid = np.load('data/valid.npy', mmap_mode='r')
        vi = np.linspace(0, len(valid)-1, 512, dtype=np.int64)
        v = torch.from_numpy(np.array(valid[vi])).cuda().permute(0, 3, 1, 2).float()/127.5-1
        vg = torch.Generator(device='cuda').manual_seed(2401)
        vt = torch.sigmoid(torch.randn(512, device='cuda', generator=vg)*.8-.8)
        vn = torch.randn(v.shape, device='cuda', generator=vg)
        va = vt[:, None, None, None]
        vx = (va*v+(1-va)*vn).to(memory_format=torch.channels_last)

    @torch.no_grad()
    def validate(m):
        m.eval()
        vals = []
        for j in range(0, 512, 32):
            pred = m(vx[j:j+32], vt[j:j+32])
            vals.append(flow_loss(pred, v[j:j+32], vt[j:j+32]).item())
        return float(np.mean(vals))

    successful = skipped = first = 0
    elapsed_before = 0.
    if args.resume_from:
        assert not args.benchmark
        resume = Path(args.resume_from)
        c = torch.load(resume/'state.pt', map_location='cpu', weights_only=False)
        assert c['world_size'] == world
        assert c['config']['global_batch'] == args.global_batch and c['config']['lr'] == args.lr
        assert c['model_config'] == model.experiment_config
        model.load_state_dict(c['model']); ema.load_state_dict(c['ema'])
        opt.load_state_dict(c['optimizer']); scaler.load_state_dict(c['scaler'])
        first = c['step']; successful = c['successful']; skipped = c['skipped']
        assert args.steps > first
        prior = json.loads((resume.parent/f'training_{first}.json').read_text())
        md['curve'] = prior['curve']
        elapsed_before = prior['elapsed_seconds']
        md['resume'] = {'directory': str(resume), 'step': first, 'elapsed_before': elapsed_before,
                        'full_optimizer_ema_scaler_rng_restored': True}
        r = torch.load(resume/f'rng_rank{rank}.pt', map_location='cpu', weights_only=False)
        assert r['step'] == first and r['rank'] == rank
        rng.bit_generator.state = r['numpy_rng']
        torch.set_rng_state(r['cpu_rng']); torch.cuda.set_rng_state(r['cuda_rng'])
        del c, r
    if rank == 0:
        print(json.dumps({'event':'start','start_step':first,'target':args.steps,'config':vars(args)}),flush=True)
    values, speed = [], []
    barrier()
    torch.cuda.synchronize()
    started = block_start = time.perf_counter()
    total = 140 if args.benchmark else args.steps
    torch.cuda.reset_peak_memory_stats()
    for step in range(first+1, total+1):
        np.copyto(host.numpy(), data[rng.integers(0, len(data), size=batch)])
        x = host.cuda(non_blocking=True).permute(0, 3, 1, 2).float()/127.5-1
        x = torch.where(torch.rand(batch, 1, 1, 1, device='cuda') < .5, x.flip(-1), x)
        x = x.to(memory_format=torch.channels_last)
        t = torch.sigmoid(torch.randn(batch, device='cuda')*.8-.8)
        a = t[:, None, None, None]
        z = a*x+(1-a)*torch.randn_like(x)
        for group in opt.param_groups:
            group['lr'] = args.lr*min(1., step*args.global_batch/16000.)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = flow_loss(wrapped(z, t), x, t)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.)
        old_scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() < old_scale:
            skipped += 1
        else:
            successful += 1
            image_updates = successful*args.global_batch/32
            decay = min(.999, (1+image_updates)/(10+image_updates))**(args.global_batch/32)
            with torch.no_grad():
                torch._foreach_mul_(ep, decay)
                torch._foreach_add_(ep, params, alpha=1-decay)
        values.append(loss.detach())
        cadence = 20 if args.benchmark else 100
        if step % cadence == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-block_start
            stats = torch.stack([torch.stack(values).mean(), torch.tensor(elapsed, device='cuda')])
            if world > 1:
                dist.all_reduce(stats[:1], op=dist.ReduceOp.SUM)
                dist.all_reduce(stats[1:], op=dist.ReduceOp.MAX)
            avg_loss, slow_time = stats.cpu().tolist()
            avg_loss /= world
            assert np.isfinite(avg_loss)
            if step > 40:
                speed.append(slow_time*1000/cadence)
            if rank == 0:
                print(json.dumps({'step': step, 'loss': avg_loss, 'ms_per_step': slow_time*1000/cadence,
                                  'images_per_second': args.global_batch*cadence/slow_time,
                                  'successful_updates': successful, 'skipped': skipped,
                                  'grad_scale': scaler.get_scale()}), flush=True)
            values = []
            block_start = time.perf_counter()
        if not args.benchmark and (step % 1250 == 0 or step == total):
            checkpoint_start = time.perf_counter()
            barrier()
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            checkpoint_dir = out/f'checkpoint_{step}'
            if rank == 0:
                checkpoint_dir.mkdir(exist_ok=True)
            barrier()
            torch.save({'step': step, 'rank': rank, 'numpy_rng': rng.bit_generator.state,
                        'cpu_rng': cpu_rng, 'cuda_rng': cuda_rng}, checkpoint_dir/f'rng_rank{rank}.pt')
            if rank == 0:
                val = validate(ema)
                sampler = FlowSampler(ema, batch=16, steps=50, graph=True)
                imgs = sampler.run(881)
                assert torch.isfinite(imgs).all()
                save_grid(imgs, out/f'ema_step{step}.png', columns=4, scale=3)
                del sampler
                gc.collect()
                torch.cuda.empty_cache()
                c = {'ema': ema.state_dict(), 'step': step, 'config': vars(args),
                     'model_config': model.experiment_config, 'world_size': world}
                torch.save(c, out/f'ema_{step}.pt')
                torch.save({**c, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
                            'scaler': scaler.state_dict(), 'successful': successful, 'skipped': skipped},
                           checkpoint_dir/'state.pt')
                row = {'step': step, 'training_image_draws': step*args.global_batch,
                       'validation_ema': {'mse': val}, 'elapsed_seconds': elapsed_before+time.perf_counter()-started,
                       'successful_updates': successful, 'skipped': skipped}
                md['curve'].append(row)
                md.update(row)
                md.update({'status': 'complete' if step == total else 'training',
                           'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20})
                write(out/'training.json', md)
                write(out/f'training_{step}.json', md)
                print(json.dumps({'event': 'checkpoint', **row}), flush=True)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            model.train()
            barrier()
            block_start += time.perf_counter() - checkpoint_start
    # Check all replicas finished with identical parameters and optimizer skip counts.
    check = torch.tensor([sum(p.detach().double().sum().item() for p in params),
                          successful, skipped], device='cuda', dtype=torch.float64)
    lo, hi = check.clone(), check.clone()
    if world > 1:
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    assert torch.equal(lo, hi), (lo, hi)
    if rank == 0:
        md.update({'replicas_agree': True, 'measured_ms': speed,
                   'median_ms': float(np.median(speed)),
                   'images_per_second': args.global_batch*1000/float(np.median(speed)),
                   'successful_updates': successful, 'skipped': skipped,
                   'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                   'scope': 'Actual RGB loading/augmentation + forward/backward/allreduce/AdamW/EMA; excludes validation and checkpoint writes.'})
        write(out/('benchmark.json' if args.benchmark else 'training.json'), md)
        print(json.dumps({'event': 'complete', 'world': world, 'step': total,
                          'images_per_second': md['images_per_second']}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
