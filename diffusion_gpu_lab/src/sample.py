"""Generate PNG samples from a saved checkpoint on CUDA."""
import argparse
from pathlib import Path
import torch
from fast import new_model, Sampler
from benchmark import save_grid

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='results/checkpoint_epoch20.pt')
    p.add_argument('--method',choices=['ddpm','ddim'],default='ddpm');p.add_argument('--steps',type=int,default=200)
    p.add_argument('--batch',type=int,default=5);p.add_argument('--seed',type=int,default=881)
    p.add_argument('--amp',action='store_true');p.add_argument('--output',default='results/sample.png')
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    m=new_model().cuda().eval();m.load_state_dict(torch.load(a.checkpoint,weights_only=True)['state_dict'])
    s=Sampler(m,batch=a.batch,method=a.method,steps=a.steps,graph=True,amp=a.amp,
              channels_last=a.amp,cache_time=True)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    save_grid(s.run(seed=a.seed),a.output)
    print(a.output)

if __name__=='__main__':main()
