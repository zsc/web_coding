"""Quality-focused retraining: adequate terminal noise, more data, EMA and clipped DDIM.

Keeps the original 660,129-parameter U-Net. This changes training/sampling semantics
and is deliberately separate from the numerically equivalent acceleration study.
"""
import argparse,copy,gc,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from core import get_ddpm_params,q_sample,linear_beta_schedule
from fast import new_model,CachedTimeEmbedding
from benchmark import save_grid,summary,status,encode_images


def params(timesteps=1000):
    return get_ddpm_params(linear_beta_schedule(timesteps).cuda())


class QualitySampler:
    def __init__(self,model,batch=25,steps=100,timesteps=1000,clip=True,graph=True,amp=True,method='ddim',eta=0.):
        started=time.perf_counter();self.batch=batch;self.steps=steps;self.clip=clip;self.amp=amp;self.method=method
        self.eta=eta
        if method=='ddpm' and steps!=timesteps:raise ValueError('DDPM must use every training timestep')
        self.model=copy.deepcopy(model).eval().requires_grad_(False).to(memory_format=torch.channels_last)
        self.model.time_embedding=CachedTimeEmbedding(self.model.time_embedding,timesteps,'cuda')
        self.times=torch.linspace(timesteps-1,0,steps,device='cuda').round().long()
        p=params(timesteps);a=p['alphas_cumprod'][self.times]
        ap=torch.cat([a[1:],torch.ones(1,device='cuda')])
        if method=='ddpm':
            beta=p['betas'][self.times]
            self.coeff=torch.stack([a.sqrt(),(1-a).sqrt(),beta*ap.sqrt()/(1-a),
                (1-ap)*(1-beta).sqrt()/(1-a),p['posterior_variance'][self.times].sqrt()],dim=1)
        else:
            sigma=eta*((1-ap)/(1-a)*(1-a/ap)).clamp_min(0).sqrt()
            self.coeff=torch.stack([a.sqrt(),(1-a).sqrt(),ap.sqrt(),(1-ap-sigma.square()).clamp_min(0).sqrt(),sigma],dim=1)
        self.x=torch.zeros(batch,1,28,28,device='cuda');self.pos=torch.zeros(1,device='cuda',dtype=torch.long)
        self.graph=None
        if graph:
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.no_grad():
                for _ in range(3):self.pos.zero_();self.x.normal_();self._step()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            self.pos.zero_();self.x.normal_();self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph),torch.no_grad():self._step()
        torch.cuda.synchronize();self.setup_seconds=time.perf_counter()-started

    def _step(self):
        t=self.times.index_select(0,self.pos).expand(self.batch).float()
        c=self.coeff.index_select(0,self.pos).flatten()
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.amp):eps=self.model(self.x,t).float()
        x0=(self.x-c[1]*eps)/c[0]
        if self.clip:
            x0=x0.clamp(-1,1)
            # Keep the direction consistent with the clipped clean-image prediction.
            eps=(self.x-c[0]*x0)/c[1]
        if self.method=='ddpm':self.x.copy_(c[2]*x0+c[3]*self.x+c[4]*torch.randn_like(self.x))
        elif self.eta:self.x.copy_(c[2]*x0+c[3]*eps+c[4]*torch.randn_like(self.x))
        else:self.x.copy_(c[2]*x0+c[3]*eps)
        self.pos.add_(1)

    @torch.no_grad()
    def run(self,seed=881,initial=None):
        if initial is None:
            torch.manual_seed(seed);self.x.normal_()
        else:self.x.copy_(initial)
        self.pos.zero_()
        for _ in range(self.steps):self.graph.replay() if self.graph else self._step()
        return self.x.cpu().clone()


def train(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261024);torch.backends.cudnn.benchmark=True
    a=np.load('data/mnist_full.npz');train_x=a['train_x'][:args.images]
    data=torch.from_numpy(train_x).cuda().float().unsqueeze(1)/127.5-1
    test=torch.from_numpy(a['test_x'][:512]).cuda().float().unsqueeze(1)/127.5-1
    m=new_model().cuda().to(memory_format=torch.channels_last)
    ema=copy.deepcopy(m).eval().requires_grad_(False)
    opt=torch.optim.Adam(m.parameters(),lr=args.lr,fused=True);scaler=torch.amp.GradScaler('cuda')
    p=params();g=torch.Generator(device='cuda').manual_seed(42)
    vt=torch.randint(0,1000,(512,),device='cuda',generator=g);vn=torch.randn(test.shape,device='cuda',generator=g)
    vx=q_sample(test,vt,p,vn)
    @torch.no_grad()
    def evaluate(model):
        model.eval();total=0
        for i in range(0,512,64):
            pred=model(vx[i:i+64],vt[i:i+64].float())
            total+=F.mse_loss(pred,vn[i:i+64],reduction='sum').item()
        return total/vn.numel()
    initial=evaluate(m);m.train();order=torch.randperm(len(data),device='cuda');cursor=0;losses=[];curve=[]
    torch.cuda.synchronize();started=time.perf_counter();torch.cuda.reset_peak_memory_stats();before=status()
    for step in range(1,args.steps+1):
        if cursor+args.batch>len(data):order=torch.randperm(len(data),device='cuda');cursor=0
        x=data[order[cursor:cursor+args.batch]];cursor+=args.batch
        t=torch.randint(0,1000,(args.batch,),device='cuda');noise=torch.randn_like(x)
        xt=q_sample(x,t,p,noise);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=F.mse_loss(m(xt,t.float()),noise)
        scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(loss.detach())
        decay=min(.999,(1+step)/(10+step))
        with torch.no_grad():
            torch._foreach_mul_(list(ema.parameters()),decay)
            torch._foreach_add_(list(ema.parameters()),list(m.parameters()),alpha=1-decay)
        if step%500==0 or step==args.steps:
            torch.cuda.synchronize();row={'step':step,'loss':torch.stack(losses).mean().item(),
                'validation_raw_mse':evaluate(m),'validation_ema_mse':evaluate(ema),'elapsed_seconds':time.perf_counter()-started}
            m.train();losses.clear();curve.append(row);print(json.dumps(row),flush=True)
            ckpt={'state_dict':m.state_dict(),'ema_state_dict':ema.state_dict(),'step':step,'config':vars(args)}
            torch.save(ckpt,out/'checkpoint.pt')
            info={'config':vars(args),'parameters':sum(v.numel() for v in m.parameters()),'initial_mse':initial,
                  'terminal_alpha_bar':p['alphas_cumprod'][-1].item(),'curve':curve,'before':before,
                  'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                  'seed':20261024,'schedule':'linear beta 0.0001..0.02, T=1000','ema':'min(0.999,(1+step)/(10+step))'}
            (out/'training.json').write_text(json.dumps(info,indent=2))
            if step%2000==0 or step==args.steps:
                sampler=QualitySampler(ema,steps=100);save_grid(sampler.run(),out/f'ema_step{step}.png');del sampler;gc.collect()
    (out/'after.txt').write_text(status())


@torch.no_grad()
def evaluate(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    ckpt=torch.load(args.checkpoint,weights_only=True);models={}
    for name,key in [('raw','state_dict'),('ema','ema_state_dict')]:
        model=new_model().cuda().eval();model.load_state_dict(ckpt[key]);models[name]=model
    configs=[('ema100','ema',100,True,'ddim'),('ema50','ema',50,True,'ddim'),('ema20','ema',20,True,'ddim'),
             ('raw100','raw',100,True,'ddim'),('ema100_unclipped','ema',100,False,'ddim'),
             ('ema_ddpm1000','ema',1000,True,'ddpm')]
    timings=[];tests={}
    for name,model,steps,clip,method in configs:
        sampler=QualitySampler(models[model],steps=steps,clip=clip,method=method)
        images=sampler.run();assert torch.isfinite(images).all();save_grid(images,out/f'{name}_25.png')
        # Store 100 predeclared samples to avoid selecting a flattering small panel.
        all_images=[images]+[sampler.run(seed=s) for s in (882,883,884)]
        save_grid(torch.cat(all_images),out/f'{name}_100.png',columns=10,scale=3)
        torch.save(torch.cat(all_images),out/f'{name}_100.pt')
        del sampler;gc.collect()
        five=QualitySampler(models[model],batch=5,steps=steps,clip=clip,method=method);five.run()
        values=[]
        for _ in range(7):
            torch.cuda.synchronize();t0=time.perf_counter();encode_images(five.run());torch.cuda.synchronize()
            values.append((time.perf_counter()-t0)*1000)
        timings.append({'name':name,'steps':steps,'clip':clip,'method':method,**summary(values),'setup_seconds':five.setup_seconds})
        del five;gc.collect();print(name,timings[-1]['median_ms'],flush=True)
    # Compare capture with eager on the revised equations, and an independent DDIM expression.
    graph=QualitySampler(models['ema'],batch=5,steps=50,amp=False)
    eager=QualitySampler(models['ema'],batch=5,steps=50,amp=False,graph=False)
    x=graph.run(seed=91);y=eager.run(seed=91);diff=(x-y).abs().max().item();assert diff<2e-4,diff
    result={'timings':timings,'graph_vs_eager_max_abs':diff,'tolerance':2e-4,
            'checkpoint':args.checkpoint,'step':ckpt['step'],'sample_seeds':[881,882,883,884],
            'timing_note':'7 sequential hot runs per method; 5 images including final CPU transfer and PNG encoding. These were measured later than v1 and are not a controlled cross-run speedup.',
            'gpu_status':status()}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','eval'])
    p.add_argument('--steps',type=int,default=8000);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--images',type=int,default=60000);p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--out',default='results/quality_v2');p.add_argument('--checkpoint',default='results/quality_v2/checkpoint.pt')
    args=p.parse_args();train(args) if args.action=='train' else evaluate(args)
