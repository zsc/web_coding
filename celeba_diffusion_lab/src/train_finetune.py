"""Controlled40k->60k RGB U-Net fine-tuning; fixed versus lower LR, paired EMA horizons."""
import argparse, copy, gc, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from model import gpu_status, save_grid
from model_flow import new_flow_model, FlowSampler, flow_loss


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--batch',type=int,default=64); p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--finetune-lr',type=float,required=True); p.add_argument('--resume',action='store_true'); p.add_argument('--out',default='results/run1')
    p.add_argument('--dc-weight',type=float,default=0.); p.add_argument('--dc-t-min',type=int,default=800)
    p.add_argument('--grayscale',action='store_true')
    p.add_argument('--variant',choices=('unet','jit'),required=True)
    args=p.parse_args(); assert args.dc_weight==0; out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(20261024); np_rng=np.random.default_rng(20261024)
    torch.backends.cudnn.benchmark=True
    suffix='_gray' if args.grayscale else ''
    data=np.load('data/train'+suffix+'.npy',mmap_mode='r'); channels=data.shape[-1]
    # Fixed evenly distributed validation indices, never from the training partition.
    valid=np.load('data/valid'+suffix+'.npy',mmap_mode='r'); vi=np.linspace(0,len(valid)-1,512,dtype=np.int64)
    v=torch.from_numpy(np.array(valid[vi])).cuda().permute(0,3,1,2).float()/127.5-1
    vg=torch.Generator(device='cuda').manual_seed(2401)
    vt=torch.sigmoid(torch.randn(512,device='cuda',generator=vg)*.8-.8); vn=torch.randn(v.shape,device='cuda',generator=vg)
    va=vt[:,None,None,None]; vx=(va*v+(1-va)*vn).to(memory_format=torch.channels_last)
    assert channels==(1 if args.grayscale else 3)
    model=new_flow_model(args.variant,channels).cuda().to(memory_format=torch.channels_last)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4,fused=True)
    scaler=torch.amp.GradScaler('cuda')
    curve=[]; elapsed_before=0.; first=0; train_seconds=0.; successful=0; skipped=0
    ckpt_path=out/'checkpoint.pt'
    if ckpt_path.exists() and not args.resume:
        raise FileExistsError('Existing checkpoint: use --resume or a new --out directory')
    if args.resume:
        c=torch.load(ckpt_path,map_location='cpu',weights_only=False)
        assert c['config']['batch']==args.batch and c['config']['lr']==args.lr
        assert c['config'].get('variant','baseline')==args.variant
        assert c['config'].get('grayscale',False)==args.grayscale
        model.load_state_dict(c['model']); ema.load_state_dict(c['ema']); opt.load_state_dict(c['optimizer'])
        scaler.load_state_dict(c['scaler']); first=c['step']; curve=c['curve']; elapsed_before=c['elapsed_seconds']
        train_seconds=c['train_seconds']; successful=c['successful']; skipped=c['skipped']
        np_rng.bit_generator.state=c['numpy_rng']; torch.set_rng_state(c['cpu_rng']); torch.cuda.set_rng_state(c['cuda_rng'])
        if args.steps<=first:
            print(json.dumps({'event':'already_complete','step':first})); return
    assert args.resume and first>=40000 and args.variant=='unet' and args.batch==32 and args.finetune_lr>0
    slow_ema=copy.deepcopy(ema).eval().requires_grad_(False)
    if first>40000:
        assert c['config']['finetune_lr']==args.finetune_lr
        slow_ema.load_state_dict(c['slow_ema'])
    metadata={'config':vars(args),'parameters':sum(p.numel() for p in model.parameters()),'seed':20261024,
              'data_manifest':json.loads(Path('data/manifest.json').read_text()),
              'gray_manifest':json.loads(Path('data/gray_manifest.json').read_text()) if args.grayscale else None,
              'torch':torch.__version__,
              'model':model.experiment_config,'schedule':'z=t*x+(1-t)*noise; t=sigmoid(N(-0.8,0.8^2)); noise_scale=1',
              'loss':'MSE((x_pred-x)/max(1-t,0.05)); equivalent to velocity-space MSE',
              'prediction':'direct clean x; no input-output skip; no clipping during sampling',
              'normalization':'same fixed uint8/127.5-1 as epsilon baseline (not JiT dataset mean/std)',
              'optimizer_note':'same AdamW/lr/warmup/EMA as epsilon study, not the original large JiT recipe',
              'ema':'Original0.999 and additional0.9999; both start from same40k EMA, update only on successful optimizer steps',
              'finetune':{'start':40000,'lr':args.finetune_lr,'slow_ema_decay':.9999,'initial_slow_ema':'same40k EMA'},
              'augmentation':'independent random horizontal flip, p=0.5',
              'sampling':'uniform training examples with replacement; no validation/test training',
              'validation':'512 fixed evenly spaced validation examples; fixed Gaussian noise and logit-normal t; FP32 evaluation',
              'before':gpu_status(),'curve':curve}
    @torch.no_grad()
    def evaluate(m):
        m.eval(); losses=[]
        for j in range(0,512,32):
            pred=m(vx[j:j+32],vt[j:j+32])
            losses.append(((pred-v[j:j+32])/(1-vt[j:j+32,None,None,None]).clamp_min(.05)).square().mean((1,2,3)))
        losses=torch.cat(losses)
        return {'mse':losses.mean().item(),'bins':[losses[(vt>=a/10)&(vt<(a+1)/10)].mean().item() if ((vt>=a/10)&(vt<(a+1)/10)).any() else None for a in range(10)]}
    if not args.resume:
        metadata['initial_validation']=evaluate(model)
    else:
        prior=json.loads((out/'training.json').read_text()); metadata['initial_validation']=prior['initial_validation']
    model.train(); host=torch.empty(args.batch,64,64,channels,dtype=torch.uint8,pin_memory=True)
    started=time.perf_counter(); bucket=[]; base_bucket=[]; torch.cuda.reset_peak_memory_stats()
    params=list(model.parameters()); ema_params=list(ema.parameters()); slow_params=list(slow_ema.parameters())
    print(json.dumps({'event':'start','step':first,'target':args.steps,'config':vars(args),'gpu':gpu_status()}),flush=True)
    for step in range(first+1,args.steps+1):
        tick=time.perf_counter()
        indices=np_rng.integers(0,len(data),size=args.batch)
        np.copyto(host.numpy(),data[indices])
        x=host.cuda(non_blocking=True).permute(0,3,1,2).float()/127.5-1
        flips=torch.rand(args.batch,1,1,1,device='cuda')<.5
        x=torch.where(flips,x.flip(-1),x).to(memory_format=torch.channels_last)
        t=torch.sigmoid(torch.randn(args.batch,device='cuda')*.8-.8); noise=torch.randn_like(x)
        a=t[:,None,None,None]; xt=a*x+(1-a)*noise
        lr=args.finetune_lr
        for group in opt.param_groups: group['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            pred=model(xt,t)
            base_loss=flow_loss(pred,x,t);loss=base_loss
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        old_scale=scaler.get_scale(); scaler.step(opt); scaler.update()
        if scaler.get_scale()<old_scale:
            skipped+=1
        else:
            successful+=1; decay=min(.999,(1+successful)/(10+successful))
            with torch.no_grad():
                torch._foreach_mul_(ema_params,decay); torch._foreach_add_(ema_params,params,alpha=1-decay)
                torch._foreach_mul_(slow_params,.9999); torch._foreach_add_(slow_params,params,alpha=.0001)
        bucket.append(loss.detach()); base_bucket.append(base_loss.detach()); train_seconds+=time.perf_counter()-tick
        if step%100==0:
            torch.cuda.synchronize()
            print(json.dumps({'step':step,'loss':torch.stack(bucket).mean().item(),
                              'elapsed_seconds':elapsed_before+time.perf_counter()-started,
                              'successful_updates':successful,'skipped':skipped,'grad_scale':scaler.get_scale()}),flush=True)
        if step%1000==0 or step==args.steps:
            torch.cuda.synchronize(); raw=evaluate(model); avg=evaluate(ema); slow_avg=evaluate(slow_ema); model.train()
            row={'step':step,'train_loss':torch.stack(bucket).mean().item(),
                 'train_velocity_mse':torch.stack(base_bucket).mean().item(),
                 'validation_raw':raw,'validation_ema':avg,'validation_slow_ema':slow_avg,'elapsed_seconds':elapsed_before+time.perf_counter()-started,
                 'successful_updates':successful,'skipped':skipped}
            curve.append(row); bucket=[]; base_bucket=[]
            # Sampling must not perturb the resumed training random stream.
            cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state()
            if step%2000==0 or step==1000 or step==args.steps:
                sampler=FlowSampler(ema,batch=16,steps=50,graph=True)
                images=sampler.run(seed=881); assert torch.isfinite(images).all()
                save_grid(images,out/f'ema_step{step}.png',columns=4,scale=3)
                del sampler; gc.collect(); torch.cuda.empty_cache()
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
            elapsed=elapsed_before+time.perf_counter()-started
            checkpoint={'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),
                        'slow_ema':slow_ema.state_dict(),'scaler':scaler.state_dict(),'step':step,'config':vars(args),'curve':curve,
                        'elapsed_seconds':elapsed,'train_seconds':train_seconds,'successful':successful,'skipped':skipped,
                        'numpy_rng':np_rng.bit_generator.state,'cpu_rng':cpu_rng,'cuda_rng':cuda_rng}
            torch.save(checkpoint,out/'checkpoint.tmp'); (out/'checkpoint.tmp').replace(ckpt_path)
            if step%5000==0:
                torch.save({'ema':ema.state_dict(),'step':step,'config':vars(args)},out/f'ema_{step}.pt')
                torch.save({'ema':slow_ema.state_dict(),'step':step,'config':vars(args),'ema_decay':.9999},out/f'slow_ema_{step}.pt')
                torch.save({'step':step,'numpy_rng':np_rng.bit_generator.state,'cuda_rng':cuda_rng},out/f'rng_{step}.pt')
            metadata.update({'curve':curve,'step':step,'elapsed_seconds':elapsed,'train_seconds':train_seconds,
                             'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                             'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                             'successful_updates':successful,'skipped':skipped,'status':'complete' if step==args.steps else 'training',
                             'current_gpu':gpu_status(),'training_image_draws':step*args.batch})
            atomic_json(out/'training.json',metadata)
            if step%5000==0: atomic_json(out/f'training_{step}.json',metadata)
            print(json.dumps({'event':'checkpoint',**row}),flush=True)
    print(json.dumps({'event':'complete','step':args.steps,'seconds':elapsed}),flush=True)

if __name__=='__main__': main()
