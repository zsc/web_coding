"""Fixed-global-batch U-Net DDP benchmark with FP32 gradient reference."""
import argparse, copy, gc, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model_flow import new_flow_model, flow_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    if world>1:dist.init_process_group('nccl')
    def barrier():
        if world>1:dist.barrier()
    model=new_flow_model('unet',3).cuda().to(memory_format=torch.channels_last)
    path=Path('results/rgb_flow_unet5k/checkpoint.pt');c=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(c['model']);ema=copy.deepcopy(model).eval().requires_grad_(False);ema.load_state_dict(c['ema'])
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True) if world>1 else model
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4,fused=True);opt.load_state_dict(c['optimizer'])
    scaler=torch.amp.GradScaler('cuda');scaler.load_state_dict(c['scaler'])
    rng=np.random.default_rng(2901);data=np.load('data/train.npy',mmap_mode='r');batches=[];h=hashlib.sha256()
    for _ in range(16):
        x=np.array(data[rng.integers(len(data),size=32)]).transpose(0,3,1,2).astype('float32')/127.5-1
        t=(1/(1+np.exp(-(rng.standard_normal(32)*.8-.8)))).astype('float32')
        z=t[:,None,None,None]*x+(1-t[:,None,None,None])*rng.standard_normal(x.shape).astype('float32')
        for v in (x,t,z):h.update(v.tobytes())
        batches.append(tuple(torch.from_numpy(v).cuda().to(memory_format=torch.channels_last) if v.ndim==4 else torch.from_numpy(v).cuda() for v in (x,t,z)))
    span=slice(rank*(32//world),(rank+1)*(32//world));x,t,z=batches[0]
    opt.zero_grad(set_to_none=True);pred=ddp(z[span],t[span]);flow_loss(pred,x[span],t[span]).backward()
    numeric={}
    if rank==0:
        ref=copy.deepcopy(model).train();ref.zero_grad(set_to_none=True);flow_loss(ref(z,t),x,t).backward()
        grads=torch.cat([p.grad.flatten() for p in model.parameters()]);expected=torch.cat([p.grad.flatten() for p in ref.parameters()])
        numeric={'gradient_relative_l2':float((grads-expected).norm()/expected.norm()),'gradient_max_abs':float((grads-expected).abs().max())}
        assert numeric['gradient_relative_l2']<2e-3,numeric
        del ref,grads,expected;gc.collect();torch.cuda.empty_cache()
    barrier();ps=list(model.parameters());ep=list(ema.parameters());skips=0;values=[]
    for block in range(7):
        barrier();torch.cuda.synchronize();start=time.perf_counter()
        for i in range(20):
            x,t,z=batches[(block*20+i)%16];opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=flow_loss(ddp(z[span],t[span]),x[span],t[span])
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(ps,1.)
            before=scaler.get_scale();scaler.step(opt);scaler.update()
            if scaler.get_scale()<before:skips+=1
            else:
                with torch.no_grad():torch._foreach_mul_(ep,.999);torch._foreach_add_(ep,ps,alpha=.001)
        torch.cuda.synchronize();duration=torch.tensor(time.perf_counter()-start,device='cuda')
        if world>1:dist.all_reduce(duration,op=dist.ReduceOp.MAX)
        if block>=2:values.append(float(duration)*1000/20)
        if rank==0:print(json.dumps({'block':block,'ms_per_step':float(duration)*1000/20,'skips':skips}),flush=True)
    if rank==0:
        r={'world_size':world,'global_batch':32,'local_batch':32//world,'median_ms':float(np.median(values)),
           'values_ms':values,'images_per_second':32000/float(np.median(values)),'skips':skips,
           'numerical':numeric,'weights_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
           'inputs_sha256':h.hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'scope':'Same resumed5k UNet state, resident RGB batch32, FP16, backward/allreduce/clip/AdamW/EMA;40 warmup then5x20 steps.'}
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'benchmark.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
