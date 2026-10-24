"""Per-rank training graphs with explicit, averaged FP32 gradient all-reduce.

Collectives stay outside graph capture. Scaled gradients are reduced before
unscale/clip, so an overflow on any rank skips the optimizer on every rank.
"""
import copy
import hashlib
import os
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from model_flow import new_flow_model, flow_loss
from model_scaled import ScaledJiT
from training_acceleration import GraphBackward


def init():
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    return rank, world


class AverageGradients:
    def __init__(self, params):
        self.params = params
        self.flat = torch.empty(sum(p.numel() for p in params), device='cuda', dtype=torch.float32)
        self.views = []
        offset = 0
        for p in params:
            self.views.append(self.flat[offset:offset+p.numel()].view(p.shape))
            offset += p.numel()

    def __call__(self):
        grads = [p.grad for p in self.params]
        assert all(g is not None and g.dtype == torch.float32 for g in grads)
        torch._foreach_copy_(self.views, grads)
        dist.all_reduce(self.flat, op=dist.ReduceOp.SUM)
        self.flat.div_(dist.get_world_size())
        torch._foreach_copy_(grads, self.views)


class TrainingState:
    def __init__(self, args, acceleration=True):
        self.args = args
        self.rank = dist.get_rank()
        self.batch = args.global_batch // dist.get_world_size()
        torch.manual_seed(args.seed)
        self.model = (ScaledJiT() if args.kind == 'scaled' else new_flow_model('unet', 3)).cuda().to(memory_format=torch.channels_last)
        c = torch.load(args.source, map_location='cpu', weights_only=False)
        self.model.load_state_dict(c['model'])
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.ema.load_state_dict(c['ema'])
        self.slow = copy.deepcopy(self.ema)
        if 'slow_ema' in c:
            self.slow.load_state_dict(c['slow_ema'])
        self.replaced_norms = 0
        if acceleration and args.kind == 'scaled':
            from triton_rmsnorm import install
            self.replaced_norms = install(self.model)
            assert self.replaced_norms == 25
        self.params = list(self.model.parameters())
        self.ep, self.sp = list(self.ema.parameters()), list(self.slow.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=args.lr, weight_decay=1e-4, fused=True)
        self.opt.load_state_dict(c['optimizer'])
        for group in self.opt.param_groups:
            group['lr'] = args.lr
        self.scaler = torch.amp.GradScaler('cuda')
        self.scaler.load_state_dict(c['scaler'])
        self.source_step = c['step']
        source_batch = c['config'].get('global_batch', c['config'].get('batch'))
        self.source_images = c.get('training_image_draws', c['step'] * source_batch)
        self.source_successful, self.source_skips = c['successful'], c['skipped']
        self.source_has_slow = 'slow_ema' in c
        del c
        self.successful = self.skipped = 0
        self.rng = np.random.default_rng(args.seed + self.rank)
        torch.manual_seed(args.seed + 1000 + self.rank)
        self.data = np.load('data/train.npy', mmap_mode='r')
        assert self.data.shape == (162770, 64, 64, 3)
        self.host = torch.empty(self.batch, 64, 64, 3, dtype=torch.uint8, pin_memory=True)
        self.graph = None
        if acceleration:
            cpu, cuda, numpy = torch.get_rng_state(), torch.cuda.get_rng_state(), copy.deepcopy(self.rng.bit_generator.state)
            self.graph = GraphBackward(self.model, self.opt, self.scaler, self.inputs())
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(cuda)
            self.rng.bit_generator.state = numpy
        self.reducer = AverageGradients(self.params)

    def inputs(self):
        np.copyto(self.host.numpy(), self.data[self.rng.integers(len(self.data), size=self.batch)])
        x = self.host.cuda(non_blocking=True).permute(0, 3, 1, 2).float()/127.5-1
        x = torch.where(torch.rand(self.batch, 1, 1, 1, device='cuda') < .5, x.flip(-1), x).to(memory_format=torch.channels_last)
        t = torch.sigmoid(torch.randn(self.batch, device='cuda')*.8-.8)
        a = t[:, None, None, None]
        return x, t, a*x + (1-a)*torch.randn_like(x)

    def update(self, batch=None, inject_inf=False):
        x, t, z = self.inputs() if batch is None else batch
        if self.graph is not None:
            loss = self.graph(x, t, z).detach().clone()
        else:
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                loss = flow_loss(self.model(z, t), x, t)
            self.scaler.scale(loss).backward()
            loss = loss.detach()
        if inject_inf and self.rank == 0:
            self.params[0].grad[(0,)*self.params[0].ndim] = float('inf')
        self.reducer()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.params, 1.)
        old = self.scaler.get_scale()
        self.scaler.step(self.opt)
        self.scaler.update()
        skipped = self.scaler.get_scale() < old
        self.skipped += int(skipped)
        self.successful += int(not skipped)
        if not skipped:
            # Same EMA time constant in image draws as the batch32 runs.
            for dst, decay in ((self.ep, .999), (self.sp, .9999)):
                decay = decay ** (self.args.global_batch / 32)
                with torch.no_grad():
                    torch._foreach_mul_(dst, decay)
                    torch._foreach_add_(dst, self.params, alpha=1-decay)
        return loss

    def verify_replicas(self):
        h = hashlib.sha256()
        for p in self.params + self.ep + self.sp:
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        for p in self.params:
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                h.update(self.opt.state[p][key].detach().cpu().contiguous().numpy().tobytes())
        record = (h.hexdigest(), self.scaler.state_dict(), self.successful, self.skipped)
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, record)
        assert all(r == records[0] for r in records), 'Replicas or scaler/optimizer states diverged'
        return {'sha256_model_dual_ema_optimizer': h.hexdigest(), 'replicas': len(records), 'identical': True}
