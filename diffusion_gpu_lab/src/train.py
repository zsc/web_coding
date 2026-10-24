"""Train the original 665k-class U-Net objective with AMP and GPU-resident data."""
import argparse, json, time, platform, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from core import q_sample
from fast import new_model, schedule
from train_graph import GraphTrainer


def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--seed',type=int,default=20261024); p.add_argument('--out',default='results')
    p.add_argument('--mode',choices=['amp','graph'],default='amp')
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    data=np.load('data/mnist_subset.npz')
    x=torch.from_numpy(data['train_x']).to('cuda').float().unsqueeze(1)/127.5-1
    x=x.contiguous(memory_format=torch.channels_last)
    test=torch.from_numpy(data['test_x'][:512]).to('cuda').float().unsqueeze(1)/127.5-1
    m=new_model().cuda()
    if a.mode=='amp':m.to(memory_format=torch.channels_last)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,fused=True)
    scaler=torch.amp.GradScaler('cuda'); params=schedule()
    g=torch.Generator(device='cuda').manual_seed(42)
    vt=torch.randint(0,200,(len(test),),device='cuda',generator=g)
    vn=torch.randn(test.shape,device='cuda',generator=g)
    vx=q_sample(test,vt,params,vn)
    @torch.no_grad()
    def evaluate():
        m.eval(); loss=0
        for start in range(0,len(test),64):
            pred=m(vx[start:start+64],vt[start:start+64].float())
            loss+=F.mse_loss(pred,vn[start:start+64],reduction='sum').item()
        m.train(); return loss/vn.numel()
    records=[]; initial_loss=evaluate()
    trainer=GraphTrainer(m) if a.mode=='graph' else None
    torch.cuda.synchronize(); started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(a.epochs):
        order=torch.randperm(len(x),device='cuda'); losses=[]
        for start in range(0,len(x)-63,64):
            batch=x[order[start:start+64]]
            t=torch.randint(0,200,(64,),device='cuda'); noise=torch.randn_like(batch)
            if trainer is not None:
                loss=trainer.step(batch,t,noise)
                # Graph output is reused on replay; snapshot before the next update.
                losses.append(loss.detach().clone())
            else:
                noisy=q_sample(batch,t,params,noise); opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    pred=m(noisy,t.float()); loss=F.mse_loss(pred,noise)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                losses.append(loss.detach())
        torch.cuda.synchronize()
        row={'epoch':epoch+1,'step':(epoch+1)*(len(x)//64),'loss':torch.stack(losses).mean().item(),
             'validation_noise_mse':evaluate(),'seconds':time.perf_counter()-started}
        records.append(row); print(json.dumps(row),flush=True)
        if epoch+1 in (5,a.epochs):
            torch.save({'state_dict':m.state_dict(),'epoch':epoch+1,'seed':a.seed},out/f'checkpoint_epoch{epoch+1}.pt')
        result={'config':vars(a),'train_images':len(x),'validation_images':len(test),'batch':64,
                'timesteps':200,'learning_rate':1e-4,'precision':'FP32 CUDA Graph' if trainer else 'AMP fp16, fp32 master weights',
                'graph_setup_seconds':trainer.setup_seconds if trainer else None,
                'parameters':sum(v.numel() for v in m.parameters()),'initial_validation_mse':initial_loss,
                'curve':records,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                'gpu':torch.cuda.get_device_name(),'host':platform.node(),
                'gpu_status':subprocess.check_output(['nvidia-smi'],text=True)}
        (out/'training.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
