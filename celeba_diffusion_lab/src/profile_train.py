"""Isolated RGB training workload: actual preprocessing, backward, AdamW and EMA.

Never writes training checkpoints. Profiler runs and timing runs are separate.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from model_flow import new_flow_model, flow_loss
from model_scaled import ScaledJiT


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


class Workload:
    def __init__(self, args):
        self.args = args
        torch.set_num_threads(4)
        torch.manual_seed(20261024)
        torch.backends.cudnn.benchmark = True
        self.model = (ScaledJiT() if args.variant == 'large_jit' else new_flow_model(args.variant, 3)).cuda().to(memory_format=torch.channels_last)
        self.weights = torch.load(args.weights, map_location='cpu', weights_only=False)['ema']
        self.model.load_state_dict(self.weights)
        self.optimization = getattr(args, 'optimization', 'eager')
        self.replaced_norms = 0
        if 'triton' in self.optimization:
            assert args.variant in ('jit', 'large_jit')
            from triton_rmsnorm import install
            self.replaced_norms = install(self.model)
            assert self.replaced_norms > 0
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.params, self.avg = list(self.model.parameters()), list(self.ema.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=2e-4, weight_decay=1e-4, fused=True)
        self.scaler = torch.amp.GradScaler('cuda', init_scale=4096.)
        self.data = np.load('data/train.npy', mmap_mode='r')
        assert self.data.shape == (162770, 64, 64, 3)
        self.host = torch.empty(args.batch, 64, 64, 3, dtype=torch.uint8, pin_memory=True)
        self.rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        self.skipped = 0
        self.range = torch.profiler.record_function if args.capture == 'torch' else lambda _: contextlib.nullcontext()
        self.graph_backward = None
        if 'graph' in self.optimization:
            from training_acceleration import GraphBackward
            cpu_rng, gpu_rng, np_rng = torch.get_rng_state(), torch.cuda.get_rng_state(), copy.deepcopy(self.rng.bit_generator.state)
            self.graph_backward = GraphBackward(self.model,self.opt,self.scaler,self.inputs())
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(gpu_rng); self.rng.bit_generator.state=np_rng

    def inputs(self):
        with self.range('train/data_h2d_augment'):
            np.copyto(self.host.numpy(), self.data[self.rng.integers(len(self.data), size=self.args.batch)])
            x = self.host.cuda(non_blocking=True).permute(0, 3, 1, 2).float()/127.5-1
            x = torch.where(torch.rand(self.args.batch, 1, 1, 1, device='cuda') < .5, x.flip(-1), x).to(memory_format=torch.channels_last)
            t = torch.sigmoid(torch.randn(self.args.batch, device='cuda')*.8-.8)
            a = t[:, None, None, None]
            z = a*x+(1-a)*torch.randn_like(x)
        return x, t, z

    def backward(self, x, t, z):
        if self.graph_backward is not None:
            with self.range('train/graph_forward_backward'):
                return self.graph_backward(x,t,z)
        self.opt.zero_grad(set_to_none=True)
        with self.range('train/forward_loss'):
            with torch.autocast('cuda', dtype=torch.float16):
                loss = flow_loss(self.model(z, t), x, t)
        with self.range('train/backward'):
            self.scaler.scale(loss).backward()
        return loss

    def update(self, supplied=None, inject_inf=False):
        x, t, z = self.inputs() if supplied is None else supplied
        loss = self.backward(x, t, z)
        if inject_inf:
            self.params[0].grad[(0,)*self.params[0].grad.ndim] = float('inf')
        with self.range('train/unscale_clip'):
            self.scaler.unscale_(self.opt)
            norm = torch.nn.utils.clip_grad_norm_(self.params, 1.)
        with self.range('train/adamw_scale_sync'):
            old = self.scaler.get_scale()
            self.scaler.step(self.opt); self.scaler.update()
            skip = self.scaler.get_scale() < old
        self.skipped += skip
        with self.range('train/ema'):
            if not skip:
                with torch.no_grad():
                    torch._foreach_mul_(self.avg, .999)
                    torch._foreach_add_(self.avg, self.params, alpha=.001)
        return loss.detach(), norm.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['unet', 'jit', 'large_jit'], required=True)
    p.add_argument('--weights', required=True); p.add_argument('--out', required=True)
    p.add_argument('--batch', type=int, default=32); p.add_argument('--seed', type=int, default=240105)
    p.add_argument('--capture', choices=['none', 'torch', 'asys'], default='none')
    p.add_argument('--warmup', type=int, default=20); p.add_argument('--steps', type=int, default=20)
    p.add_argument('--blocks', type=int, default=5)
    p.add_argument('--optimization',choices=['eager','graph','triton','graph_triton'],default='eager')
    args = p.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.device_count() == 1
    w = Workload(args)
    result = {'config': vars(args), 'torch': torch.__version__, 'python': platform.python_version(),
              'device': str(torch.cuda.get_device_properties(0)), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'parameters': sum(x.numel() for x in w.params), 'task_id': os.environ.get('SOFY_TASK_ID'),
              'weights_sha256': hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'RGB64 batch32; disk cache indexing/H2D/augmentation+FP16 forward/backward+unscale/clip+fused AdamW+1 EMA; excludes validation and checkpoints; single device, no DDP'}
    result.update(replaced_norms=w.replaced_norms, graph_setup_seconds=w.graph_backward.setup_seconds if w.graph_backward else 0)
    for _ in range(args.warmup): loss, norm = w.update()
    torch.cuda.synchronize(); warm_skips = w.skipped
    torch.cuda.reset_peak_memory_stats()
    timings = []
    if args.capture == 'asys':
        torch.cuda.profiler.start()
        for _ in range(args.steps): loss, norm = w.update()
        torch.cuda.synchronize(); torch.cuda.profiler.stop()
    elif args.capture == 'torch':
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(args.steps):
                with torch.profiler.record_function('train/step'): loss, norm = w.update()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(out/'torch_trace.json'))
        (out/'torch_ops.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=50))
    else:
        for _ in range(args.blocks):
            torch.cuda.synchronize(); started = time.perf_counter()
            for _ in range(args.steps): loss, norm = w.update()
            torch.cuda.synchronize(); timings.append((time.perf_counter()-started)*1000/args.steps)
    assert torch.isfinite(loss) and torch.isfinite(norm)
    result.update(block_ms=timings, median_ms=float(np.median(timings)) if timings else None,
                  final_loss=float(loss), final_norm=float(norm), warmup_skips=warm_skips,
                  measured_skips=w.skipped-warm_skips, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    write(out/'result.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
