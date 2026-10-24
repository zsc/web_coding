"""Resumable 64px CelebA training with FP32 weights, AMP, EMA and fixed validation."""
import argparse, copy, gc, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from model import new_model, alpha_bar, gpu_status, Sampler, save_grid


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--batch',type=int,default=64); p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--resume',action='store_true'); p.add_argument('--out',default='results/run1')
    args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(20261024); np_rng=np.random.default_rng(20261024)
    torch.backends.cudnn.benchmark=True
    data=np.load('data/train.npy',mmap_mode='r')
    # Fixed evenly distributed validation indices, never from the training partition.
    valid=np.load('data/valid.npy',mmap_mode='r'); vi=np.linspace(0,len(valid)-1,512,dtype=np.int64)
    v=torch.from_numpy(np.array(valid[vi])).cuda().permute(0,3,1,2).float()/127.5-1
    vg=torch.Generator(device='cuda').manual_seed(2401)
    vt=torch.arange(512,device='cuda')*1000//512; vn=torch.randn(v.shape,device='cuda',generator=vg)
    ab=alpha_bar(); va=ab[vt,None,None,None]; vx=(va.sqrt()*v+(1-va).sqrt()*vn).to(memory_format=torch.channels_last)
    model=new_model().cuda().to(memory_format=torch.channels_last)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4,fused=True)
    scaler=torch.amp.GradScaler('cuda')
    curve=[]; elapsed_before=0.; first=0; train_seconds=0.; successful=0; skipped=0
    ckpt_path=out/'checkpoint.pt'
    if args.resume:
        c=torch.load(ckpt_path,map_location='cpu',weights_only=False)
        assert c['config']['batch']==args.batch and c['config']['lr']==args.lr
        model.load_state_dict(c['model']); ema.load_state_dict(c['ema']); opt.load_state_dict(c['optimizer'])
        scaler.load_state_dict(c['scaler']); first=c['step']; curve=c['curve']; elapsed_before=c['elapsed_seconds']
        train_seconds=c['train_seconds']; successful=c['successful']; skipped=c['skipped']
        np_rng.bit_generator.state=c['numpy_rng']; torch.set_rng_state(c['cpu_rng']); torch.cuda.set_rng_state(c['cuda_rng'])
    metadata={'config':vars(args),'parameters':sum(p.numel() for p in model.parameters()),'seed':20261024,
              'data_manifest':json.loads(Path('data/manifest.json').read_text()),'torch':torch.__version__,
              'model':dict(model.config),'schedule':'linear beta 0.0001..0.02, T=1000, epsilon MSE',
              'terminal_alpha_bar':ab[-1].item(),'ema':'min(0.999, (1+successful_updates)/(10+successful_updates))',
              'augmentation':'independent random horizontal flip, p=0.5',
              'sampling':'uniform training examples with replacement; no validation/test training',
              'validation':'512 fixed evenly spaced validation examples; fixed Gaussian noise; stratified t=0..998; FP32 evaluation',
              'before':gpu_status(),'curve':curve}
    @torch.no_grad()
    def evaluate(m):
        m.eval(); losses=[]
        for j in range(0,512,32):
            pred=m(vx[j:j+32],vt[j:j+32],return_dict=False)[0]
            losses.append((pred-vn[j:j+32]).square().mean((1,2,3)))
        losses=torch.cat(losses)
        return {'mse':losses.mean().item(),'bins':[losses[(vt>=a)&(vt<a+100)].mean().item() for a in range(0,1000,100)]}
    if not args.resume:
        metadata['initial_validation']=evaluate(model)
    else:
        prior=json.loads((out/'training.json').read_text()); metadata['initial_validation']=prior['initial_validation']
    model.train(); host=torch.empty(args.batch,64,64,3,dtype=torch.uint8,pin_memory=True)
    started=time.perf_counter(); last_tick=started; bucket=[]; torch.cuda.reset_peak_memory_stats()
    params=list(model.parameters()); ema_params=list(ema.parameters())
    print(json.dumps({'event':'start','step':first,'target':args.steps,'config':vars(args),'gpu':gpu_status()}),flush=True)
    for step in range(first+1,args.steps+1):
        tick=time.perf_counter()
        indices=np_rng.integers(0,len(data),size=args.batch)
        np.copyto(host.numpy(),data[indices])
        x=host.cuda(non_blocking=True).permute(0,3,1,2).float()/127.5-1
        flips=torch.rand(args.batch,1,1,1,device='cuda')<.5
        x=torch.where(flips,x.flip(-1),x).to(memory_format=torch.channels_last)
        t=torch.randint(0,1000,(args.batch,),device='cuda'); noise=torch.randn_like(x)
        a=ab[t,None,None,None]; xt=a.sqrt()*x+(1-a).sqrt()*noise
        lr=args.lr*min(1.,step/500)
        for group in opt.param_groups: group['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            loss=F.mse_loss(model(xt,t,return_dict=False)[0],noise)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        old_scale=scaler.get_scale(); scaler.step(opt); scaler.update()
        if scaler.get_scale()<old_scale:
            skipped+=1
        else:
            successful+=1; decay=min(.999,(1+successful)/(10+successful))
            with torch.no_grad():
                torch._foreach_mul_(ema_params,decay); torch._foreach_add_(ema_params,params,alpha=1-decay)
        bucket.append(loss.detach()); train_seconds+=time.perf_counter()-tick
        if step%100==0:
            torch.cuda.synchronize()
            print(json.dumps({'step':step,'loss':torch.stack(bucket).mean().item(),
                              'elapsed_seconds':elapsed_before+time.perf_counter()-started,
                              'successful_updates':successful,'skipped':skipped,'grad_scale':scaler.get_scale()}),flush=True)
        if step%1000==0 or step==args.steps:
            torch.cuda.synchronize(); raw=evaluate(model); avg=evaluate(ema); model.train()
            row={'step':step,'train_loss':torch.stack(bucket).mean().item(),
                 'validation_raw':raw,'validation_ema':avg,'elapsed_seconds':elapsed_before+time.perf_counter()-started,
                 'successful_updates':successful,'skipped':skipped}
            curve.append(row); bucket=[]
            # Sampling must not perturb the resumed training random stream.
            cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state()
            if step%2000==0 or step==1000 or step==args.steps:
                sampler=Sampler(ema,batch=16,steps=100,eta=0.,graph=True)
                images=sampler.run(seed=881); assert torch.isfinite(images).all()
                save_grid(images,out/f'ema_step{step}.png',columns=4,scale=3)
                del sampler; gc.collect(); torch.cuda.empty_cache()
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
            elapsed=elapsed_before+time.perf_counter()-started
            checkpoint={'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),
                        'scaler':scaler.state_dict(),'step':step,'config':vars(args),'curve':curve,
                        'elapsed_seconds':elapsed,'train_seconds':train_seconds,'successful':successful,'skipped':skipped,
                        'numpy_rng':np_rng.bit_generator.state,'cpu_rng':cpu_rng,'cuda_rng':cuda_rng}
            torch.save(checkpoint,out/'checkpoint.tmp'); (out/'checkpoint.tmp').replace(ckpt_path)
            if step%5000==0: torch.save({'ema':ema.state_dict(),'step':step,'config':vars(args)},out/f'ema_{step}.pt')
            metadata.update({'curve':curve,'step':step,'elapsed_seconds':elapsed,'train_seconds':train_seconds,
                             'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                             'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                             'successful_updates':successful,'skipped':skipped,'status':'complete' if step==args.steps else 'training',
                             'current_gpu':gpu_status(),'training_image_draws':step*args.batch})
            atomic_json(out/'training.json',metadata); print(json.dumps({'event':'checkpoint',**row}),flush=True)
    print(json.dumps({'event':'complete','step':args.steps,'seconds':elapsed}),flush=True)

if __name__=='__main__': main()
