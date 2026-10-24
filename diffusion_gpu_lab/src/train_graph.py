"""FP32 full training-step CUDA Graph, with explicit mutable batch/t/noise inputs."""
import time
import torch
import torch.nn.functional as F
from core import q_sample
from fast import schedule

class GraphTrainer:
    def __init__(self, model, batch=64, lr=1e-4):
        start=time.perf_counter();self.model=model.train()
        self.x=torch.zeros(batch,1,28,28,device='cuda')
        self.t=torch.zeros(batch,device='cuda',dtype=torch.long)
        self.noise=torch.zeros_like(self.x);self.params=schedule()
        self.opt=torch.optim.Adam(model.parameters(),lr=lr,fused=True,capturable=True)
        initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self._step()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.opt.zero_grad(set_to_none=True)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self._step()
        # Warmup must not silently count as optimizer steps. Restore values in place,
        # keeping the addresses captured by the graph intact.
        model.load_state_dict(initial)
        for state in self.opt.state.values():
            for value in state.values():
                if torch.is_tensor(value):value.zero_()
        torch.cuda.synchronize();self.setup_seconds=time.perf_counter()-start

    def _step(self):
        self.opt.zero_grad(set_to_none=True)
        pred=self.model(q_sample(self.x,self.t,self.params,self.noise),self.t.float())
        self.loss=F.mse_loss(pred,self.noise)
        self.loss.backward();self.opt.step()

    def step(self,x,t=None,noise=None):
        self.x.copy_(x)
        self.t.random_(200) if t is None else self.t.copy_(t)
        self.noise.normal_() if noise is None else self.noise.copy_(noise)
        self.graph.replay()
        return self.loss
