"""Synchronized, interleaved measurements against the original GPU implementation."""
import argparse, copy, gc, io, json, platform, random, subprocess, time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from core import q_sample, p_sample, p_sample_loop
from fast import new_model, schedule, Sampler


def status():
    return subprocess.check_output(['nvidia-smi'],text=True)


def save_grid(images, path, columns=5, scale=5):
    a=((images.detach().cpu().float().clamp(-1,1)+1)*127.5).round().to(torch.uint8).squeeze(1).numpy()
    rows=(len(a)+columns-1)//columns
    canvas=np.full((rows*32+4,columns*32+4),20,dtype=np.uint8)
    for i,im in enumerate(a):
        y,x=divmod(i,columns);canvas[4+y*32:32+y*32,4+x*32:32+x*32]=im
    Image.fromarray(canvas).resize((canvas.shape[1]*scale,canvas.shape[0]*scale),Image.Resampling.NEAREST).save(path)


def encode_images(x):
    a=((x.clamp(-1,1)+1)*127.5).to(torch.uint8).squeeze(1).numpy()
    result=[]
    for im in a:
        b=io.BytesIO();Image.fromarray(im).save(b,format='PNG');result.append(b.getvalue())
    return result


def summary(values):
    return {'median_ms':float(np.median(values)),'p10_ms':float(np.percentile(values,10)),
            'p90_ms':float(np.percentile(values,90)),'min_ms':min(values),'max_ms':max(values),'samples_ms':values}


@torch.no_grad()
def validate(m, out):
    torch.manual_seed(101);p=schedule();x=torch.randn(5,1,28,28,device='cuda')
    noises=torch.randn(200,*x.shape,device='cuda')
    t=torch.tensor([0,1,99,198,199],device='cuda');n=torch.randn_like(x)
    direct=p['alphas_cumprod'][t,None,None,None].sqrt()*x+(1-p['alphas_cumprod'][t,None,None,None]).sqrt()*n
    q_err=(q_sample(x,t,p,n)-direct).abs().max().item()
    ref=x.clone()
    for k,i in enumerate(reversed(range(200))):
        # Use the unmodified upstream step with explicitly controlled noise.
        with patch('torch.randn_like',return_value=noises[k]):
            ref=p_sample(m,ref,torch.full((5,),i,device='cuda').float(),i,p)
    equivalent=Sampler(m,cache_time=True)
    actual=equivalent.fixed_noise_run(x,noises)
    maxerr=(ref-actual).abs().max().item();rmse=(ref-actual).square().mean().sqrt().item()
    assert q_err<1e-6 and maxerr<2e-4,(q_err,maxerr)
    graph=Sampler(m,method='ddim',steps=20,graph=True,cache_time=True)
    eager=Sampler(m,method='ddim',steps=20,cache_time=True)
    g=graph.run(x);e=eager.run(x);ge=(g-e).abs().max().item()
    assert ge<2e-4,ge
    noisy_graph=Sampler(m,graph=True,cache_time=True)
    a=noisy_graph.run(x,seed=19);b=noisy_graph.run(x,seed=19);c=noisy_graph.run(x)
    repeat=(a-b).abs().max().item();fresh=(b-c).square().mean().sqrt().item()
    assert repeat<2e-4 and fresh>1e-3
    amp=Sampler(m,amp=True,channels_last=True,cache_time=True)
    ap=amp.fixed_noise_run(x,noises)
    result={'q_sample_max_abs':q_err,'ddpm_200_same_noise_max_abs':maxerr,
            'ddpm_200_same_noise_rmse':rmse,'ddim_graph_vs_eager_max_abs':ge,
            'graph_seed_replay_max_abs':repeat,'graph_fresh_noise_rmse':fresh,
            'amp_200_same_noise_max_abs':(ref-ap).abs().max().item(),
            'amp_200_same_noise_rmse':(ref-ap).square().mean().sqrt().item(),
            'fp32_tolerance':2e-4,'passed':True}
    (out/'validation.json').write_text(json.dumps(result,indent=2));print('VALIDATION',json.dumps(result),flush=True)
    save_grid(torch.cat([ref.cpu(),actual.cpu(),ap.cpu()]),out/'equivalence.png')


def training_bench(m, out):
    torch.manual_seed(123)
    cpu=torch.from_numpy(np.load('data/mnist_subset.npz')['train_x'][:64]).unsqueeze(1).float()/127.5-1
    params=schedule(); rows=[]
    for name,amp,cl,fused in [('original_fp32',False,False,False),('gpu_resident_fp32',False,True,True),('amp_channels_last',True,True,True)]:
        model=copy.deepcopy(m).train().requires_grad_(True)
        if cl:model.to(memory_format=torch.channels_last)
        resident=cpu.cuda().contiguous(memory_format=torch.channels_last if cl else torch.contiguous_format)
        opt=torch.optim.Adam(model.parameters(),lr=1e-4,fused=fused)
        scaler=torch.amp.GradScaler('cuda',enabled=amp)
        def step():
            opt.zero_grad(set_to_none=True)
            batch=resident if cl else cpu.cuda()
            t=torch.randint(0,200,(64,),device='cuda');noise=torch.randn_like(batch)
            noisy=q_sample(batch,t,params,noise)
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                loss=F.mse_loss(model(noisy,t.float()),noise)
            if not cl:loss.item() # Original app synchronizes each batch before backward.
            scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            return loss
        for _ in range(8):step()
        values=[];torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            torch.cuda.synchronize();t0=time.perf_counter()
            for _ in range(20):loss=step()
            torch.cuda.synchronize();values.append((time.perf_counter()-t0)*1000/20)
        assert torch.isfinite(loss).item()
        row={'name':name,**summary(values),'batch':64,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
        rows.append(row); print('TRAIN_BENCH',json.dumps(row),flush=True)
        del step, model,opt,scaler,resident;gc.collect();torch.cuda.empty_cache()
    (out/'training_benchmark.json').write_text(json.dumps(rows,indent=2))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='results/checkpoint_epoch20.pt')
    p.add_argument('--repeats',type=int,default=7);p.add_argument('--out',default='results')
    a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=new_model().cuda().eval();m.load_state_dict(torch.load(a.checkpoint,weights_only=True)['state_dict'])
    before=status(); validate(m,out); gc.collect();torch.cuda.empty_cache()
    params=schedule()
    def original():
        return torch.cat([p_sample_loop(m,(1,1,28,28),200,params,'cuda')[0] for _ in range(5)])
    specs=[('batch5_fp32',dict()),('graph_fp32',dict(graph=True,cache_time=True)),
           ('graph_amp',dict(graph=True,amp=True,channels_last=True,cache_time=True)),
           ('ddim50_amp',dict(method='ddim',steps=50,graph=True,amp=True,channels_last=True,cache_time=True)),
           ('ddim20_amp',dict(method='ddim',steps=20,graph=True,amp=True,channels_last=True,cache_time=True))]
    samplers={name:Sampler(m,**args) for name,args in specs}
    funcs={'original_serial5':original,**{k:v.run for k,v in samplers.items()}}
    setup={k:v.setup_seconds for k,v in samplers.items()}
    # Every method gets a full request warmup; measured repetitions are interleaved.
    for name,fn in funcs.items():
        print('WARM',name,flush=True);encode_images(fn());torch.cuda.synchronize()
    measurements={k:[] for k in funcs}; telemetry=[];rng=random.Random(73)
    for rep in range(a.repeats):
        names=list(funcs);rng.shuffle(names)
        telemetry.append({'repetition':rep,'gpu':status(),'order':names})
        for name in names:
            torch.cuda.synchronize();t0=time.perf_counter()
            encode_images(funcs[name]());torch.cuda.synchronize()
            ms=(time.perf_counter()-t0)*1000;measurements[name].append(ms)
            print('BENCH',rep,name,round(ms,3),flush=True)
    rows=[{'name':k,**summary(v),'steps':50 if '50' in k else 20 if '20' in k else 200,
           'setup_seconds':setup.get(k,0)} for k,v in measurements.items()]
    base=rows[0]['median_ms']
    for r in rows:r['speedup']=base/r['median_ms']
    result={'config':vars(a),'rows':rows,'before':before,'after':status(),'telemetry':telemetry,
            'timing_scope':'synchronized wall clock, five 28x28 images, includes RNG, sampling, final D2H and PNG encoding; excludes model loading, setup, Flask/HTTP and SSH',
            'batch':5,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
    (out/'benchmark.json').write_text(json.dumps(result,indent=2))
    torch.manual_seed(881);initial=torch.randn(5,1,28,28,device='cuda')
    gallery={}
    for name in ['graph_fp32','graph_amp','ddim50_amp','ddim20_amp']:
        images,traces=samplers[name].run(initial,seed=117,trace=True)
        save_grid(images,out/f'{name}.png');save_grid(torch.cat(traces),out/f'{name}_trajectory.png')
        gallery[name]={'min':images.min().item(),'max':images.max().item(),'finite':bool(torch.isfinite(images).all()),
                       'mean':images.mean().item(),'std':images.std().item()}
    (out/'gallery.json').write_text(json.dumps(gallery,indent=2))
    # Test fixed-noise heldout predictions on identical x_t; useful AMP check, not FID.
    test=torch.from_numpy(np.load('data/mnist_subset.npz')['test_x'][:512]).cuda().float().unsqueeze(1)/127.5-1
    torch.manual_seed(42);t=torch.randint(0,200,(len(test),),device='cuda');n=torch.randn_like(test)
    xt=q_sample(test,t,params,n);errors={}
    for name,model,amp in [('fp32',m,False),('amp',samplers['graph_amp'].model,True)]:
        total=0
        for i in range(0,len(test),64):
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):pred=model(xt[i:i+64],t[i:i+64].float()).float()
            total+=F.mse_loss(pred,n[i:i+64],reduction='sum').item()
        errors[name]=total/n.numel()
    (out/'quality.json').write_text(json.dumps({'heldout_epsilon_mse':errors,'images':512,
           'note':'No FID or generative distribution equivalence claim. DDIM uses the same checkpoint but changes the reverse process.'},indent=2))
    del samplers,funcs;gc.collect();torch.cuda.empty_cache()
    with torch.enable_grad():training_bench(m,out)

if __name__=='__main__':main()
