"""Additional graph validation and predeclared multi-seed sample gallery."""
import json,gc
from pathlib import Path
import torch
from fast import new_model,Sampler,schedule
from benchmark import save_grid

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    out=Path('results');m=new_model().cuda().eval()
    m.load_state_dict(torch.load(out/'checkpoint_epoch20.pt',weights_only=True)['state_dict'])
    e=Sampler(m,cache_time=True);g=Sampler(m,cache_time=True,graph=True)
    torch.manual_seed(881);x=torch.randn(5,1,28,28,device='cuda')
    a=e.run(x,seed=117);b=g.run(x,seed=117)
    err=(a-b).abs().max().item()
    # The device RNG sequence should match under identical seeds and tensor layout.
    assert err<2e-4,err
    # At t=0 the variance is zero, so a graph replay must equal upstream mean exactly.
    p=schedule();from core import p_sample
    expected=p_sample(m,x,torch.zeros(5,device='cuda'),0,p)
    g.x.copy_(x);g.pos.fill_(199);g.graph.replay()
    zero=(g.x-expected).abs().max().item();assert zero<2e-4,zero
    result={'ddpm_graph_vs_eager_seeded_200_max_abs':err,'ddpm_graph_t0_max_abs':zero,
            'alpha_bar_terminal':p['alphas_cumprod'][-1].item(),'threshold':2e-4,
            'gallery_initial_seeds':[881,882,883,884,885],'gallery_noise_seeds':[117,118,119,120,121]}
    (out/'additional_validation.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
    del e,g;gc.collect();torch.cuda.empty_cache()
    for name,kw in [('graph_fp32',{}),('graph_amp',dict(amp=True,channels_last=True)),
                    ('ddim50_amp',dict(amp=True,channels_last=True,method='ddim',steps=50)),
                    ('ddim20_amp',dict(amp=True,channels_last=True,method='ddim',steps=20))]:
        sampler=Sampler(m,graph=True,cache_time=True,**kw);images=[]
        for seed in range(5):
            torch.manual_seed(881+seed);initial=torch.randn(5,1,28,28,device='cuda')
            images.append(sampler.run(initial,seed=117+seed))
        save_grid(torch.cat(images),out/f'{name}_25.png');del sampler
    m.load_state_dict(torch.load(out/'checkpoint_epoch5.pt',weights_only=True)['state_dict'])
    s=Sampler(m,graph=True,cache_time=True);torch.manual_seed(881);initial=torch.randn(5,1,28,28,device='cuda')
    save_grid(s.run(initial,seed=117),out/'epoch5.png')

if __name__=='__main__':main()
