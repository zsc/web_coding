"""Visual controls: original checkpoint, clipping only, and schedule-only retraining."""
from pathlib import Path
import torch
from fast import new_model,Sampler
from quality_v2 import QualitySampler
from benchmark import save_grid

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    out=Path('results/quality_v2');m=new_model().cuda().eval()
    m.load_state_dict(torch.load('results/checkpoint_epoch20.pt',weights_only=True)['state_dict'])
    torch.manual_seed(881);initial=torch.randn(25,1,28,28,device='cuda')
    original=Sampler(m,batch=25,graph=True,cache_time=True)
    save_grid(original.run(initial=initial,seed=117),out/'original_ddpm_25.png')
    for clip in [False,True]:
        s=QualitySampler(m,batch=25,steps=100,timesteps=200,clip=clip)
        save_grid(s.run(initial=initial),out/f'old_ddim100_clip{int(clip)}_25.png')
    fixed=torch.load('results/schedule_only/checkpoint.pt',weights_only=True)
    m.load_state_dict(fixed['state_dict']);s=QualitySampler(m,batch=25,steps=100)
    save_grid(s.run(initial=initial),out/'schedule_only_25.png')

if __name__=='__main__':main()
