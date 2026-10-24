"""Opt-in exact-scope acceleration, leaving historical model/trainer sources intact."""
import time
import torch
from model_flow import flow_loss


class GraphBackward:
    """Capture only forward/loss/scaled backward; keep GradScaler/AdamW outside.

    Parameters, gradients, inputs and GradScaler scale retain their addresses.
    Each replay overwrites captured leaf gradients (zero_grad(None) at capture).
    Dynamic loss scaling, clipping, overflow skips and EMA remain unchanged.
    """
    def __init__(self, model, optimizer, scaler, example):
        started = time.perf_counter()
        self.model, self.optimizer, self.scaler = model, optimizer, scaler
        self.x, self.t, self.z = [v.detach().clone() for v in example]
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()
        self.grad_ptrs = [p.grad.data_ptr() for p in model.parameters()]
        torch.cuda.synchronize(); self.setup_seconds = time.perf_counter()-started

    def _body(self):
        # Recast current FP32 parameters on every replay, without cached stale weights.
        with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            self.loss = flow_loss(self.model(self.z, self.t), self.x, self.t)
        self.scaler.scale(self.loss).backward()

    def __call__(self, x, t, z):
        self.x.copy_(x); self.t.copy_(t); self.z.copy_(z)
        self.graph.replay()
        return self.loss
