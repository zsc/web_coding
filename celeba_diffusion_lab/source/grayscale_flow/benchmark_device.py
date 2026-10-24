"""Matched single-device training and sampling benchmark, including host overhead."""
import argparse, copy, gc, hashlib, json, platform, subprocess, time
from pathlib import Path
import numpy as np
import torch
from model_flow import new_flow_model, flow_loss, FlowSampler

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def telemetry():
    p=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
    return p.stdout or p.stderr
def summary(values):
    a=np.array(values)
    return {'median_ms':float(np.median(a)*1000),'p10_ms':float(np.percentile(a,10)*1000),
            'p90_ms':float(np.percentile(a,90)*1000),'values_ms':(a*1000).tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--out',required=True)
    p.add_argument('--weights',required=True);p.add_argument('--variant',default='jit',choices=['jit','unet'])
    p.add_argument('--batch',type=int,default=32);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261024);torch.backends.cudnn.benchmark=True
    result={'label':a.label,'torch':torch.__version__,'python':platform.python_version(),
            'device':str(torch.cuda.get_device_properties(0)),'visible_devices':torch.cuda.device_count(),
            'weights_sha256':sha(a.weights),'code_sha256':sha(__file__), 'batch':a.batch,
            'precision':'FP32 parameters/state, FP16 autocast, GradScaler, FP32 loss',
            'scope':'Resident real CelebA batches: forward + backward + clip + AdamW + EMA; excludes data I/O, validation and checkpoint writes',
            'before':telemetry()}
    assert torch.cuda.device_count()==1, 'Expose exactly one device for this benchmark'
    model=new_flow_model(a.variant).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=False)['ema'])
    result['parameters']=sum(v.numel() for v in model.parameters())
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4,fused=True)
    scaler=torch.amp.GradScaler('cuda',init_scale=4096.)
    params=list(model.parameters());avg=list(ema.parameters());skipped=0
    # NumPy creates identical inputs on both runtimes; hash verifies the actual tensors.
    rng=np.random.default_rng(240105);data=np.load('data/train_gray.npy',mmap_mode='r');batches=[];digest=hashlib.sha256()
    for _ in range(16):
        x=np.array(data[rng.integers(len(data),size=a.batch)]).transpose(0,3,1,2).astype('float32')/127.5-1
        flips=rng.random(a.batch)<.5; x[flips]=x[flips,:,:,::-1]
        t=(1/(1+np.exp(-(rng.standard_normal(a.batch)*.8-.8)))).astype('float32')
        noise=rng.standard_normal(x.shape).astype('float32');xt=t[:,None,None,None]*x+(1-t[:,None,None,None])*noise
        for v in (x,t,xt):digest.update(v.tobytes())
        batches.append((torch.from_numpy(x).cuda().to(memory_format=torch.channels_last),torch.from_numpy(t).cuda(),torch.from_numpy(xt).cuda().to(memory_format=torch.channels_last)))
    result['inputs_sha256']=digest.hexdigest()
    with torch.no_grad():
        model.eval();x,t,xt=batches[0];fp32=model(xt,t)
        with torch.autocast('cuda',dtype=torch.float16):mixed=model(xt,t)
        result['forward_amp_fp32_rmse']=float((mixed.float()-fp32).square().mean().sqrt())
        np.savez(out/'forward.npz',fp32=fp32.cpu().numpy(),amp=mixed.float().cpu().numpy())
    model.train()
    def update(i):
        nonlocal skipped
        x,t,xt=batches[i%len(batches)];opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=flow_loss(model(xt,t),x,t)
        scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(params,1.)
        before=scaler.get_scale();scaler.step(opt);scaler.update()
        if scaler.get_scale()<before:skipped+=1
        else:
            with torch.no_grad():torch._foreach_mul_(avg,.999);torch._foreach_add_(avg,params,alpha=.001)
        return loss.detach(),norm.detach()
    torch.cuda.synchronize();started=time.perf_counter()
    for i in range(32):loss,norm=update(i)
    torch.cuda.synchronize();result['warmup_seconds']=time.perf_counter()-started;warm_skips=skipped
    blocks=[];losses=[];torch.cuda.reset_peak_memory_stats()
    for block in range(5):
        torch.cuda.synchronize();started=time.perf_counter()
        for j in range(40):loss,norm=update(32+block*40+j)
        torch.cuda.synchronize();blocks.append((time.perf_counter()-started)/40)
        assert torch.isfinite(loss) and torch.isfinite(norm);losses.append(float(loss))
    result.update(training=summary(blocks),images_per_second=a.batch/float(np.median(blocks)),
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  warmup_skips=warm_skips,measured_skips=skipped-warm_skips,block_end_losses=losses)
    del opt,scaler,ema,avg,params,batches;gc.collect();torch.cuda.empty_cache()
    # Reset to exactly the same EMA weights before measuring inference.
    model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=False)['ema']);model.eval()
    result['sampling']={}
    for graph in (False,True):
        sampler=FlowSampler(model,batch=5,steps=50,graph=graph,amp=True);sampler.run(881)
        times=[]
        for i in range(7):
            torch.cuda.synchronize();started=time.perf_counter();images=sampler.run(881+i)
            torch.cuda.synchronize();times.append(time.perf_counter()-started);assert torch.isfinite(images).all()
        result['sampling']['graph' if graph else 'eager']={**summary(times),'setup_seconds':sampler.setup_seconds,'batch':5,'nfe':99,'includes':'CPU output copy; excludes PNG encoding'}
        del sampler;gc.collect();torch.cuda.empty_cache()
    result['after']=telemetry();(out/'benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
