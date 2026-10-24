"""One hundred real full-state continuation updates, historical trainer vs graph."""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
import torch
from check_training_acceleration import diff


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);src=Path(a.source)
    initial=torch.load(src/'checkpoint.pt',map_location='cpu',weights_only=False)
    target=initial['step']+100; lr=initial['config']['finetune_lr']
    for mode,script in [('eager','src/train_finetune.py'),('graph','src/train_finetune_fast.py')]:
        d=out/mode;d.mkdir();shutil.copy2(src/'checkpoint.pt',d/'checkpoint.pt');shutil.copy2(src/'training.json',d/'training.json')
        cmd=[sys.executable,'-u',script,'--variant','unet','--steps',str(target),'--batch','32','--resume','--finetune-lr',str(lr),'--out',str(d)]
        with (out/(mode+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
    ref=torch.load(out/'eager/checkpoint.pt',map_location='cpu',weights_only=False)
    fast=torch.load(out/'graph/checkpoint.pt',map_location='cpu',weights_only=False)
    checks={k:diff(list(fast[k].values()),list(ref[k].values())) for k in ('model','ema','slow_ema')}
    for k in ('step','exp_avg','exp_avg_sq'):
        checks['optimizer_'+k]=diff([s[k] for s in fast['optimizer']['state'].values()],[s[k] for s in ref['optimizer']['state'].values()])
    checks['scaler_equal']=fast['scaler']==ref['scaler']
    checks['numpy_rng_equal']=fast['numpy_rng']==ref['numpy_rng']
    checks['cpu_rng_equal']=torch.equal(fast['cpu_rng'],ref['cpu_rng'])
    checks['cuda_rng_equal']=torch.equal(fast['cuda_rng'],ref['cuda_rng'])
    checks['skip_equal']=fast['skipped']==ref['skipped']
    checks['successful_equal']=fast['successful']==ref['successful']
    result={'source':str(src),'start':initial['step'],'target':target,'checks':checks,
            'validation_eager':ref['curve'][-1],'validation_graph':fast['curve'][-1],
            'scope':'Full trained model, optimizer, dynamic scaler, both EMA states and RNG restored;100 identical real data updates plus validation/sampling/checkpoint round trip'}
    result['bitwise_equal']=all(v['max_abs']==0 if isinstance(v,dict) else v for v in checks.values())
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    assert result['bitwise_equal']


if __name__=='__main__':main()
