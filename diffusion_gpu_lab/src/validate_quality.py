"""Oracle-noise checks against closed-form forward diffusion at sampled times."""
import json
from pathlib import Path
import torch
from fast import new_model
from quality_v2 import QualitySampler,params

class Oracle(torch.nn.Module):
    def __init__(self,noise):super().__init__();self.noise=noise
    def forward(self,x,t):return self.noise

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(52);m=new_model().cuda().eval();p=params()
    x0=torch.rand(5,1,28,28,device='cuda')*1.6-.8;eps=torch.randn_like(x0);tests=[]
    for eta in [0.,1.]:
        s=QualitySampler(m,batch=5,steps=50,eta=eta,graph=False,amp=False)
        s.model=Oracle(eps)
        for k in [0,25,49]:
            t=int(s.times[k].item());prev=int(s.times[k+1].item()) if k<49 else -1
            a=p['alphas_cumprod'][t];ap=p['alphas_cumprod'][prev] if prev>=0 else torch.tensor(1.,device='cuda')
            xt=a.sqrt()*x0+(1-a).sqrt()*eps;s.x.copy_(xt);s.pos.fill_(k)
            torch.manual_seed(66);s._step()
            torch.manual_seed(66);z=torch.randn_like(x0)
            sigma=eta*((1-ap)/(1-a)*(1-a/ap)).clamp_min(0).sqrt()
            expected=ap.sqrt()*x0+(1-ap-sigma.square()).clamp_min(0).sqrt()*eps+sigma*z
            error=(s.x-expected).abs().max().item();assert error<2e-4,(eta,k,error)
            tests.append({'eta':eta,'step_index':k,'max_abs':error})
    Path('results/quality_v2/oracle_validation.json').write_text(json.dumps({'tests':tests,'threshold':2e-4,'passed':True},indent=2));print(tests)

if __name__=='__main__':main()
