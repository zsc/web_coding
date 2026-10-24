"""Fixed RGB/KID evaluation of both EMAs after an accelerated training milestone."""
import argparse, gc, hashlib, json
from pathlib import Path
import numpy as np
import torch
from eval_protocol import load_model, assess
from evaluate_rgb_flow import write


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--step',type=int,required=True)
    a=p.parse_args();root=Path(a.root);out=root/'evaluation';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    path=out/'evaluation.json'
    result=json.loads(path.read_text()) if path.exists() else {'rows':[],'protocol':'RGB64;512 fixed samples seeds881..912;Heun50/99NFE;2048 real;KID50 subsets256 seed42;subsetSD not CI.'}
    real=np.load('results/rgb_flow_evaluation/real_features.npy')
    from pytorch_fid.inception import InceptionV3
    net=InceptionV3([3]).cuda().eval().requires_grad_(False)
    for prefix in ('ema','slow_ema'):
        name=f'{root.name}_{a.step}_{prefix}'
        if any(r['name']==name for r in result['rows']):continue
        weights=root/f'{prefix}_{a.step}.pt';model,c=load_model('unet',weights)
        assert c['step']==a.step
        row=assess(model,out,name,net,real)
        row.update(run=root.name,step=a.step,batch=32,weights_prefix=prefix,weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                   training=json.loads((root/f'training_{a.step}.json').read_text()))
        result['rows'].append(row);result.update(last_evaluated_step=a.step,status='complete' if prefix=='slow_ema' else 'evaluating')
        write(path,result)
        print(json.dumps({k:v for k,v in row.items() if k not in ('training','values')}),flush=True)
        del model,c;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
