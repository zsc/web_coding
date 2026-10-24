"""CelebA epsilon-prediction DDPM, trained from scratch (no pretrained weights)."""
import copy
import io
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import UNet2DModel


class GlobalAffineUNet(UNet2DModel):
    """Image-derived conditioning enters every residual scale/shift via temb.

    Identity class embedding is only an API transport for a continuous vector;
    no labels or clean-image statistics are supplied to the model.
    """
    def __init__(self, **kwargs):
        super().__init__(class_embed_type='identity', **kwargs)
        channels=self.config.in_channels; dim=self.config.block_out_channels[0]*4
        self.global_encoder=torch.nn.Sequential(
            torch.nn.Conv2d(channels,16,3,padding=1),torch.nn.SiLU(),
            torch.nn.Conv2d(16,32,3,stride=2,padding=1),torch.nn.SiLU())
        self.global_project=torch.nn.Sequential(
            torch.nn.Linear(32+2*channels,dim),torch.nn.SiLU(),torch.nn.Linear(dim,dim))
        # A zero residual condition preserves the paired time-affine model at init.
        torch.nn.init.zeros_(self.global_project[-1].weight)
        torch.nn.init.zeros_(self.global_project[-1].bias)
        self.register_to_config(global_mean_scale=64.,global_condition='raw mean/std + unnormalized conv pooled features')

    def global_condition(self, sample):
        raw=sample.float()
        mean=raw.mean((2,3)); std=raw.var((2,3),correction=0).add(1e-6).sqrt()
        pooled=self.global_encoder(sample).float().mean((2,3))
        # At pure unit Gaussian noise, 64*mean has unit SD on a 64x64 image.
        return self.global_project(torch.cat((64*mean,std,pooled),dim=1))

    def forward(self, sample, timestep, return_dict=True):
        return super().forward(sample,timestep,class_labels=self.global_condition(sample),return_dict=return_dict)


def new_model(channels=3, variant='baseline'):
    if variant not in ('baseline','time_affine','global_affine'):
        raise ValueError(f'Unknown model variant: {variant}')
    cls=GlobalAffineUNet if variant=='global_affine' else UNet2DModel
    model=cls(
        sample_size=64, in_channels=channels, out_channels=channels, layers_per_block=2,
        block_out_channels=(32, 64, 128, 128),
        down_block_types=('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D', 'DownBlock2D'),
        up_block_types=('UpBlock2D', 'AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D'),
        norm_num_groups=8, attention_head_dim=32,
        resnet_time_scale_shift='default' if variant=='baseline' else 'scale_shift',
    )
    model.register_to_config(variant=variant)
    return model


def alpha_bar(device='cuda'):
    return torch.cumprod(1 - torch.linspace(1e-4, .02, 1000, device=device), dim=0)


def gpu_status():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu,power.draw',
                                    '--format=csv'], text=True).strip()


def save_grid(x, path, columns=8, scale=2):
    a=((x.detach().cpu().float().clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).numpy()
    if a.shape[-1]==1: a=np.repeat(a,3,axis=-1)
    h,w=a.shape[1:3]; gap=2; rows=(len(a)+columns-1)//columns
    canvas=np.full((rows*(h+gap)+gap,columns*(w+gap)+gap,3),24,np.uint8)
    for i,im in enumerate(a):
        y=(i//columns)*(h+gap)+gap; z=(i%columns)*(w+gap)+gap
        canvas[y:y+h,z:z+w]=im
    image=Image.fromarray(canvas)
    if scale!=1: image=image.resize((image.width*scale,image.height*scale),Image.Resampling.NEAREST)
    image.save(path)


def png_bytes(x):
    arrays=((x.clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).numpy()
    result=[]
    for a in arrays:
        b=io.BytesIO(); Image.fromarray(a).save(b,format='PNG'); result.append(b.getvalue())
    return result


class Sampler:
    """One captured denoising step; dynamic position and state remain on the GPU."""
    def __init__(self, model, batch=16, steps=100, eta=0., graph=True, amp=True, clip=True, correct_eps=True):
        start=time.perf_counter()
        self.model=model.eval(); self.batch=batch; self.steps=steps; self.eta=eta; self.amp=amp
        self.clip=clip; self.correct_eps=correct_eps
        self.times=torch.linspace(999,0,steps,device='cuda').round().long()
        a=alpha_bar()[self.times]; ap=torch.cat((a[1:],torch.ones(1,device='cuda')))
        sigma=eta*((1-ap)/(1-a)*(1-a/ap)).clamp_min(0).sqrt()
        self.coeff=torch.stack((a.sqrt(),(1-a).sqrt(),ap.sqrt(),
                               (1-ap-sigma.square()).clamp_min(0).sqrt(),sigma),dim=1)
        self.x=torch.zeros(batch,model.config.in_channels,64,64,device='cuda').to(memory_format=torch.channels_last)
        self.pos=torch.zeros(1,device='cuda',dtype=torch.long); self.graph=None
        if graph:
            stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.no_grad():
                for _ in range(3): self.pos.zero_(); self.x.normal_(); self._step()
            torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
            self.pos.zero_(); self.x.normal_(); self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph),torch.no_grad(): self._step()
        torch.cuda.synchronize(); self.setup_seconds=time.perf_counter()-start

    def _step(self):
        t=self.times.index_select(0,self.pos).expand(self.batch)
        c=self.coeff.index_select(0,self.pos).flatten()
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.amp):
            eps=self.model(self.x,t,return_dict=False)[0].float()
        x0=(self.x-c[1]*eps)/c[0]
        if self.clip: x0=x0.clamp(-1,1)
        if self.clip and self.correct_eps: eps=(self.x-c[0]*x0)/c[1]
        nxt=c[2]*x0+c[3]*eps
        if self.eta: nxt=nxt+c[4]*torch.randn_like(self.x)
        self.x.copy_(nxt); self.pos.add_(1)

    @torch.no_grad()
    def run(self, seed=881, initial=None):
        torch.manual_seed(seed)
        if initial is None:
            self.x.normal_()
        else: self.x.copy_(initial)
        self.pos.zero_()
        for _ in range(self.steps):
            if self.graph: self.graph.replay()
            else: self._step()
        return self.x.cpu().clone()
