"""Transfer interval autoguidance from local diffusion_demo2 to RGB64 controls."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from eval_protocol import load_model, assess
from model_flow import FlowSampler
from evaluate_rgb_flow import write


class Guided(torch.nn.Module):
    def __init__(self, strong, weak, scale, start=.1, end=.8):
        super().__init__()
        self.strong, self.weak = strong, weak
        self.scale, self.start, self.end = scale, start, end
        self.in_channels = 3

    def forward(self, x, t):
        good = self.strong(x,t)
        if self.scale == 1:
            return good
        bad = self.weak(x,t)
        active = ((t>=self.start)&(t<=self.end))[:,None,None,None]
        # Tensor-only interval mask is compatible with CUDA Graph capture.
        return good.float()+(self.scale-1)*active*(good.float()-bad.float())


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('unet','jit'),required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    root=Path('results/rgb_flow_unet40k' if args.kind=='unet' else 'results/rgb_flow_jit20k')
    weakroot=Path('results/rgb_flow_unet5k' if args.kind=='unet' else 'results/rgb_flow_jit20k')
    strongpath,weakpath=root/'ema_20000.pt',weakroot/'ema_5000.pt'
    strong,c=load_model(args.kind,strongpath);weak,w=load_model(args.kind,weakpath)
    out=Path('results')/f'rgb_guidance_{args.kind}';out.mkdir(parents=True,exist_ok=True)
    a=FlowSampler(strong,batch=4,steps=50);ref=a.run(3341)
    del a;gc.collect();torch.cuda.empty_cache()
    a=FlowSampler(Guided(strong,weak,1),batch=4,steps=50);same=a.run(3341)
    assert torch.equal(ref,same), 'Scale1 must exactly reproduce the original trajectory'
    del a;gc.collect();torch.cuda.empty_cache()
    x=torch.randn(3,3,64,64,device='cuda');t=torch.tensor([0.,.5,.9],device='cuda')
    good=strong(x,t);test=Guided(strong,weak,1.6)(x,t)
    assert torch.equal(good[[0,2]],test[[0,2]]), 'Outside interval must preserve predictions'
    result={'kind':args.kind,'rows':[], 'checks':{'scale1_trajectory_bitwise_equal':True,'inactive_interval_exact':True},
            'strong_sha256':hashlib.sha256(strongpath.read_bytes()).hexdigest(),
            'weak_sha256':hashlib.sha256(weakpath.read_bytes()).hexdigest(),
            'strong_step':20000,'weak_step':5000,
            'source':'Inspired by /Users/zsc/d/diffusion_demo2/sample_pixel.py; https://arxiv.org/abs/2406.02507 and https://arxiv.org/abs/2404.07724',
            'protocol':'Original RGB64; identical512 samples, seeds881..912, Heun50/99 evaluations, same2048 real features. Interval[.1,.8], t=0 noise to1 data.',
            'cost':'This graph implementation executes both networks at every evaluation, including inactive times:198 model forwards versus99 baseline. No speedup is claimed.',
            'warning':'Compare full unselected grids for stripes/distortion; lower KID alone is insufficient.'}
    from pytorch_fid.inception import InceptionV3
    net=InceptionV3([3]).cuda().eval().requires_grad_(False)
    real=np.load('results/rgb_flow_evaluation/real_features.npy')
    for scale in (1.3,1.6,2.):
        name=f'{args.kind}_g{scale:.1f}'
        row=assess(Guided(strong,weak,scale),out,name,net,real)
        row.update(scale=scale,interval=[.1,.8])
        result['rows'].append(row);write(out/'evaluation.json',result)
        print(json.dumps({k:v for k,v in row.items() if k!='values'}),flush=True)
    result['status']='complete';write(out/'evaluation.json',result)


if __name__=='__main__':main()
