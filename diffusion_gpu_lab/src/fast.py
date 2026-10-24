"""Batched DDPM/DDIM and a fixed-shape, one-step CUDA Graph sampler.

The U-Net topology and training objective are unchanged. DDIM is explicitly an
alternative sampler, not numerically equivalent to the original DDPM chain.
"""
import copy
import time
import torch
from torch import nn
from core import UNet, linear_beta_schedule, get_ddpm_params


def new_model():
    return UNet(image_channels=1, n_channels=32, ch_mults=(1, 2), time_emb_dim=128)


def schedule(device='cuda', steps=200):
    return get_ddpm_params(linear_beta_schedule(steps).to(device))


class CachedTimeEmbedding(nn.Module):
    def __init__(self, original, timesteps, device):
        super().__init__()
        with torch.no_grad():
            self.register_buffer('table', original(torch.arange(timesteps, device=device).float()))

    def forward(self, t):
        return self.table[t.long()]


def inference_model(model, channels_last=False, cache_time=False):
    m = copy.deepcopy(model).eval().requires_grad_(False)
    if channels_last:
        m.to(memory_format=torch.channels_last)
    if cache_time:
        m.time_embedding = CachedTimeEmbedding(m.time_embedding, 200, next(m.parameters()).device)
    return m


class Sampler:
    """One mutable sampler per worker. Call run serially; it returns owned CPU output.

    The graph consumes a fresh Philox RNG draw on every replay. All 200 DDPM
    timesteps remain. Timesteps, coefficients and image state stay on the GPU.
    Setup/capture cost is reported separately from hot request latency.
    """
    def __init__(self, model, batch=5, method='ddpm', steps=200,
                 graph=False, amp=False, channels_last=False, cache_time=False):
        started = time.perf_counter()
        self.model = inference_model(model, channels_last, cache_time)
        self.batch, self.method, self.steps = batch, method, steps
        self.amp, self.device = amp, next(model.parameters()).device
        if method == 'ddpm' and steps != 200:
            raise ValueError('DDPM must retain all 200 trained timesteps; select DDIM for fewer steps')
        if not 2 <= steps <= 200:
            raise ValueError('steps must be in [2, 200]')
        p = schedule(self.device)
        self.times = torch.linspace(199, 0, steps, device=self.device).round().long()
        a = p['alphas_cumprod'][self.times]
        if method == 'ddpm':
            beta = p['betas'][self.times]
            c = (torch.rsqrt(1-beta), beta/torch.sqrt(1-a),
                 p['posterior_variance'][self.times].sqrt())
        elif method == 'ddim':
            ap = torch.cat([a[1:], torch.ones(1, device=self.device)])
            # x_prev = c0*x + c1*epsilon, eta=0, no x0 clipping.
            c = ((ap/a).sqrt(), (1-ap).sqrt()-(ap/a).sqrt()*(1-a).sqrt(), torch.zeros_like(a))
        else:
            raise ValueError(method)
        self.coeff = torch.stack(c, 1).contiguous()
        fmt = torch.channels_last if channels_last else torch.contiguous_format
        self.x = torch.empty((batch, 1, 28, 28), device=self.device).contiguous(memory_format=fmt)
        self.pos = torch.zeros(1, device=self.device, dtype=torch.long)
        self.graph = None
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):
                    self.pos.zero_(); self.x.normal_(); self._step()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.pos.zero_(); self.x.normal_()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), torch.no_grad():
                self._step()
        torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter()-started

    def _step(self):
        t = self.times.index_select(0, self.pos).expand(self.batch).float()
        c = self.coeff.index_select(0, self.pos).flatten()
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
            eps = self.model(self.x, t)
        eps = eps.float()
        if self.method == 'ddpm':
            self.x.copy_(c[0]*(self.x-c[1]*eps)+c[2]*torch.randn_like(self.x))
        else:
            self.x.copy_(c[0]*self.x+c[1]*eps)
        self.pos.add_(1)

    @torch.no_grad()
    def run(self, initial=None, seed=None, trace=False):
        if seed is not None:
            torch.manual_seed(seed)
        if initial is None:
            self.x.normal_()
        else:
            self.x.copy_(initial)
        self.pos.zero_()
        snapshots = []
        for i in range(self.steps):
            self.graph.replay() if self.graph is not None else self._step()
            if trace and (i % max(1, self.steps//8) == 0 or i == self.steps-1):
                snapshots.append(self.x.cpu().clone())
        result = self.x.cpu().clone()
        return (result, snapshots) if trace else result

    @torch.no_grad()
    def fixed_noise_run(self, initial, noises):
        """Validation path: same weights, initial image and per-step noise as reference."""
        x = initial.clone()
        for k in range(self.steps):
            t = self.times[k].expand(self.batch).float()
            with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
                eps = self.model(x, t).float()
            c = self.coeff[k]
            if self.method == 'ddpm':
                x = c[0]*(x-c[1]*eps)+c[2]*noises[k]
            else:
                x = c[0]*x+c[1]*eps
        return x
