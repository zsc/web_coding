"""Matched U-Net / local-reference JiT models for direct clean-x flow training."""
from dataclasses import asdict
import time
import torch
from model import new_model
from vendor_jit.config import ModelConfig,ShapeSpec
from vendor_jit.models.jit_stft import JiTSTFT

class FlowUNet(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=new_model(1,'global_affine')
        # Match direct clean-x zero output at initialization, as in the JiT source.
        torch.nn.init.zeros_(self.net.conv_out.weight);torch.nn.init.zeros_(self.net.conv_out.bias)
        self.experiment_config={'architecture':'global_affine_unet','unet':dict(self.net.config),
                                'time_direction':'0=noise, 1=data','time_embedding_scale':1000.,'zero_output_init':True}
    def forward(self,x,t):
        return self.net(x,1000*t,return_dict=False)[0]

class FlowJiT(JiTSTFT):
    def __init__(self):
        cfg=ModelConfig(in_channels=1,patch_size=(4,4),depth=9,hidden_dim=192,num_heads=6,
                        bottleneck_dim=32,output_conv_layers=4,output_conv_hidden=64)
        shapes=ShapeSpec(64,64,64,64,4,4,16,16,1)
        super().__init__(cfg,shapes)
        self.experiment_config={'architecture':'JiTSTFT (exact local reference implementation)',
                                'model':asdict(cfg),'shapes':asdict(shapes)}

def new_flow_model(kind):
    if kind=='unet':return FlowUNet()
    if kind=='jit':return FlowJiT()
    raise ValueError(kind)

def flow_loss(pred,x,t):
    # Algebraically equal to MSE(v_pred,v_target); FP32 reduction.
    return ((pred.float()-x.float())/(1-t[:,None,None,None]).clamp_min(.05)).square().mean()

class FlowSampler:
    """FP32 Heun integration, 50 intervals / 99 model evaluations by default.

    No intermediate clipping or per-image normalization. The last interval uses
    Euler; a final clamp occurs only in the existing uint8 display/metric codec.
    """
    def __init__(self,model,batch=16,steps=50,graph=True,amp=True):
        started=time.perf_counter();self.model=model.eval();self.batch=batch;self.steps=steps;self.amp=amp
        self.x=torch.zeros(batch,1,64,64,device='cuda').to(memory_format=torch.channels_last)
        self.t=torch.zeros(batch,device='cuda');self.dt=torch.tensor(1/steps,device='cuda')
        self.graph=None
        if graph:
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.no_grad():
                for _ in range(3):self.x.normal_();self.t.zero_();self._heun()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            self.x.normal_();self.t.zero_();self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph),torch.no_grad():self._heun()
        torch.cuda.synchronize();self.setup_seconds=time.perf_counter()-started
    def velocity(self,x,t):
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.amp):pred=self.model(x,t).float()
        return (pred-x)/(1-t[:,None,None,None]).clamp_min(.05)
    def _heun(self):
        v=self.velocity(self.x,self.t)
        v2=self.velocity(self.x+self.dt*v,self.t+self.dt)
        self.x.add_(.5*self.dt*(v+v2))
    @torch.no_grad()
    def run(self,seed=881,initial=None):
        torch.manual_seed(seed)
        if initial is None:self.x.normal_()
        else:self.x.copy_(initial)
        for i in range(self.steps-1):
            self.t.fill_(i/self.steps)
            if self.graph:self.graph.replay()
            else:self._heun()
        self.t.fill_((self.steps-1)/self.steps)
        self.x.add_(self.dt*self.velocity(self.x,self.t))
        return self.x.cpu().clone()
