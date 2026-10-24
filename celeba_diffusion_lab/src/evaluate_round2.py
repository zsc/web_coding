"""Evaluate each scheduled RGB checkpoint once it becomes available."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from eval_protocol import load_model, assess
from evaluate_rgb_flow import write


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=2);args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    tasks=[]
    for name,batch in [('jit_b128_lr2e4',128),('jit_b128_lr8e4',128),('jit_b512_lr2e4',512),('jit_b512_lr8e4',512)]:
        for images in (1280000,2560000):
            if name=='jit_b128_lr2e4' and images==1280000:continue
            tasks.append({'run':name,'kind':'scaled','root':f'results/rgb_round2/{name}',
                          'step':images//batch,'weights_prefix':'ema','batch':batch})
    tasks.append({'run':'small_jit40k','kind':'jit','root':'results/rgb_flow_jit40k',
                  'step':40000,'weights_prefix':'ema','batch':32})
    for name in ('unet_ft_lr2e4','unet_ft_lr5e5'):
        for step in (50000,60000):
            for prefix in ('ema','slow_ema'):
                tasks.append({'run':name,'kind':'unet','root':f'results/rgb_round2/{name}',
                              'step':step,'weights_prefix':prefix,'batch':32})
    pending=tasks[args.shard::args.shards]
    out=Path(f'results/rgb_round2_evaluation_{args.shard}');out.mkdir(parents=True,exist_ok=True)
    result={'status':'waiting','rows':[], 'pending':pending,
            'protocol':'RGB64;512fixed samples seeds881..912;Heun50/99NFE;2048 real reference;KID50 subsets256 seed42;subsetSD is not CI.'}
    write(out/'evaluation.json',result)
    real=np.load('results/rgb_flow_evaluation/real_features.npy')
    from pytorch_fid.inception import InceptionV3
    net=InceptionV3([3]).cuda().eval().requires_grad_(False)
    deadline=time.monotonic()+7200
    while pending:
        if time.monotonic()>deadline:raise TimeoutError(f'Missing checkpoints:{pending}')
        ready=next((x for x in pending if (Path(x['root'])/f"training_{x['step']}.json").exists()),None)
        if ready is None:time.sleep(5);continue
        root=Path(ready['root']);step=ready['step'];weights=root/f"{ready['weights_prefix']}_{step}.pt"
        model,c=load_model(ready['kind'],weights)
        assert c['step']==step
        name=f"{ready['run']}_{step}_{ready['weights_prefix']}"
        row=assess(model,out,name,net,real)
        row.update(ready);row['weights_sha256']=hashlib.sha256(weights.read_bytes()).hexdigest()
        row['training']=json.loads((root/f'training_{step}.json').read_text())
        result['rows'].append(row);pending.remove(ready)
        result.update(status='complete' if not pending else 'waiting',pending=pending)
        write(out/'evaluation.json',result)
        print(json.dumps({k:v for k,v in row.items() if k not in ('training','values')}),flush=True)
        del model,c;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
