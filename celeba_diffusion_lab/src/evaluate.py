"""Fixed-seed galleries, interleaved hot latency, and optional Inception KID."""
import argparse, gc, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from model import new_model, Sampler, save_grid, png_bytes, gpu_status


def stats(values):
    return {'median_ms':float(np.median(values)), 'p10_ms':float(np.percentile(values,10)),
            'p90_ms':float(np.percentile(values,90)), 'all_ms':values}


def load_models(path):
    c=torch.load(path,map_location='cpu',weights_only=False); models={}
    for name in ('model','ema'):
        m=new_model().cuda().to(memory_format=torch.channels_last).eval().requires_grad_(False)
        m.load_state_dict(c[name]); models[name]=m
    return c,models


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='results/run1/checkpoint.pt')
    p.add_argument('--out',default='results/evaluation'); p.add_argument('--kid',action='store_true')
    args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    c,models=load_models(args.checkpoint); before=gpu_status()
    configs=[('ema100','ema',100,0.),('ema50','ema',50,0.),('ema20','ema',20,0.),
             ('ema100_eta1','ema',100,1.),('ema50_eta1','ema',50,1.),('raw100','model',100,0.),
             ('ema1000_eta1','ema',1000,1.)]
    galleries=[]
    for name,key,steps,eta in configs:
        sampler=Sampler(models[key],batch=16,steps=steps,eta=eta)
        ims=torch.cat([sampler.run(s) for s in (881,882,883,884)])
        assert torch.isfinite(ims).all()
        save_grid(ims,out/(name+'_64.png'),columns=8,scale=2)
        torch.save(ims,out/(name+'_64.pt')); galleries.append(name)
        print(json.dumps({'gallery':name}),flush=True); del sampler; gc.collect(); torch.cuda.empty_cache()
    # Hot five-image latency. Randomly interleave methods to reduce drift from the shared GPU.
    benchmark_configs=[('eager_fp32_50',50,False,False,0.),('graph_fp32_50',50,True,False,0.),
                       ('graph_amp_50',50,True,True,0.),('graph_amp_20',20,True,True,0.),('graph_amp_100',100,True,True,0.),
                       ('graph_amp_50_eta1',50,True,True,1.),('graph_amp_100_eta1',100,True,True,1.),
                       ('eager_fp32_1000_eta1',1000,False,False,1.)]
    samplers={name:Sampler(models['ema'],batch=5,steps=steps,graph=graph,amp=amp,eta=eta)
              for name,steps,graph,amp,eta in benchmark_configs}
    for s in samplers.values(): s.run(91)
    exact=samplers['graph_fp32_50'].run(91); reference=samplers['eager_fp32_50'].run(91)
    maxerr=(exact-reference).abs().max().item(); assert maxerr<3e-4,maxerr
    amp=samplers['graph_amp_50'].run(91)
    amp_rmse=(amp-exact).square().mean().sqrt().item()
    values={k:[] for k in samplers}; order=[]; rng=random.Random(42)
    for rep in range(7):
        names=list(samplers); rng.shuffle(names)
        for name in names:
            torch.cuda.synchronize(); start=time.perf_counter(); png_bytes(samplers[name].run(881+rep)); torch.cuda.synchronize()
            values[name].append((time.perf_counter()-start)*1000); order.append(name)
    timings=[{'name':name,'steps':steps,'graph':graph,'amp':amp,'eta':eta,'setup_seconds':samplers[name].setup_seconds,**stats(values[name])}
             for name,steps,graph,amp,eta in benchmark_configs]
    result={'step':c['step'],'checkpoint':args.checkpoint,'galleries':galleries,'sample_seeds':[881,882,883,884],
            'timings':timings,'timing_order':order,'timing_note':'7 randomly interleaved hot runs per method, batch=5, including GPU-to-CPU and PNG encoding; excludes model load and graph capture. Shared GPU.',
            'graph_vs_eager_max_abs_fp32':maxerr,'amp_vs_fp32_trajectory_rmse':amp_rmse,
            'before':before,'after':gpu_status()}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
    del samplers; gc.collect(); torch.cuda.empty_cache()
    if args.kid: run_kid(models['ema'],out,c['step'])


@torch.no_grad()
def run_kid(model,out,step):
    from pytorch_fid.inception import InceptionV3
    # Offline cached weights from the pytorch-fid upstream release. These are evaluator weights, not generator weights.
    feature_net=InceptionV3([3]).cuda().eval().requires_grad_(False)
    def features(x):
        result=[]
        for i in range(0,len(x),32):
            # InceptionV3 internally resizes to 299 and normalizes [0,1] to [-1,1].
            f=feature_net(x[i:i+32].cuda().float())[0].flatten(1)
            result.append(f.cpu().numpy())
        return np.concatenate(result)
    valid=np.load('data/valid.npy',mmap_mode='r'); indices=np.linspace(0,len(valid)-1,2048,dtype=np.int64)
    real=features(torch.from_numpy(np.array(valid[indices])).permute(0,3,1,2).float()/255)
    np.save(out/'validation_inception_features.npy',real)
    rows=[]
    for name,steps,eta in [('ema100',100,0.),('ema50',50,0.),('ema100_eta1',100,1.),('ema50_eta1',50,1.)]:
        start=time.perf_counter(); sampler=Sampler(model,batch=16,steps=steps,eta=eta)
        generated=[]
        for j in range(64):
            x=sampler.run(seed=881+j).clamp(-1,1)
            # Match the quantized PNG representation used for viewing and the uint8 reference images.
            generated.append(features(((x+1)*127.5).round()/255))
        fake=np.concatenate(generated); np.save(out/(name+'_inception_features.npy'),fake)
        rng=np.random.default_rng(42); values=[]; n=256; d=real.shape[1]
        for _ in range(50):
            a=real[rng.choice(len(real),n,replace=False)].astype(np.float64)
            b=fake[rng.choice(len(fake),n,replace=False)].astype(np.float64)
            aa=(a@a.T/d+1)**3; bb=(b@b.T/d+1)**3; ab=(a@b.T/d+1)**3
            values.append(float((aa.sum()-aa.trace()+bb.sum()-bb.trace())/(n*(n-1))-2*ab.mean()))
        row={'name':name,'kid_mean':float(np.mean(values)),'subset_std':float(np.std(values,ddof=1)),
             'values':values,'seconds':time.perf_counter()-start}
        rows.append(row); print(json.dumps({'kid':row}),flush=True)
        del sampler; gc.collect(); torch.cuda.empty_cache()
        result={'step':step,'real_count':2048,'generated_count_per_method':1024,'subset_size':256,'subsets':50,
                'reference':'official validation split, evenly spaced indices; test split untouched',
                'features':'pytorch-fid 0.3.0 TensorFlow-compatible Inception pool3 2048d; upstream pretrained evaluator only',
                'estimator':'unbiased squared MMD with polynomial kernel (x dot y / 2048 + 1)^3; 50 fixed-seed random subsets',
                'note':'Small pilot estimate. Subset standard deviation is not a confidence interval; not comparable with published full-dataset FID.',
                'results':rows}
        (out/'kid.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
