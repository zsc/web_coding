"""Numerical reference, real-model gradients, and memory/throughput checks."""
import argparse,gc,json,time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from model_flow import new_flow_model,FlowSampler,flow_loss
from vendor_jit.flow.sampler import sample_ode

p=argparse.ArgumentParser();p.add_argument('--channels',type=int,choices=(1,3),default=1);p.add_argument('--out',default='results/flow_validation.json');args=p.parse_args();channels=args.channels
torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.manual_seed(20261024)
class Analytic(torch.nn.Module):
    def forward(self,z,t,condition=None,**kw):return .2*z.sin()+.1*t[:,None,None,None]
m=Analytic();m.in_channels=channels;cfg=SimpleNamespace(shapes=SimpleNamespace(freq_pad=64,time_pad=64),
    model=SimpleNamespace(in_channels=channels,num_classes=0),flow=SimpleNamespace(t_eps=.05,noise_scale=1.),
    sample=SimpleNamespace(num_steps=50,method='heun',cfg_scale=1.,cfg_interval=(0.1,1.)))
x=torch.randn(4,channels,64,64,device='cuda').to(memory_format=torch.channels_last)
sam=FlowSampler(m,batch=4,steps=50,graph=False,amp=False);a=sam.run(initial=x)
b=sample_ode(m,cfg,torch.ones(1,channels,64,64,device='cuda'),4,device='cuda',z_init=x)
reference_error=(a-b.cpu()).abs().max().item();assert reference_error<2e-5
sam=FlowSampler(m,batch=4,steps=50,graph=True,amp=False);c=sam.run(initial=x)
graph_error=(a-c).abs().max().item();assert graph_error<2e-5
z=torch.randn_like(x);t=torch.rand(4,device='cuda');target=torch.randn_like(x);denom=(1-t[:,None,None,None]).clamp_min(.05)
explicit=(((z-x)/denom-(target-x)/denom)**2).mean();compact=flow_loss(z,target,t)
loss_error=abs(explicit.item()-compact.item());assert torch.allclose(explicit,compact,atol=1e-5,rtol=1e-6)
result={'channels':channels,'reference_heun_max_abs_error':reference_error,'graph_heun_max_abs_error':graph_error,'velocity_loss_algebra_abs_error':loss_error,'models':{}}
del sam;gc.collect();torch.cuda.empty_cache()
for kind in ('unet','jit'):
    torch.manual_seed(20261024);model=new_flow_model(kind,channels).cuda().to(memory_format=torch.channels_last)
    with torch.no_grad():assert model(x,t).abs().max()==0
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4,fused=True);scaler=torch.amp.GradScaler('cuda')
    data=torch.randn(32,channels,64,64,device='cuda').clamp(-1,1).to(memory_format=torch.channels_last)
    times=[];losses=[];torch.cuda.reset_peak_memory_stats()
    for i in range(12):
        ts=torch.sigmoid(torch.randn(32,device='cuda')*.8-.8);n=torch.randn_like(data);v=ts[:,None,None,None];xt=v*data+(1-v)*n
        opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.float16):loss=flow_loss(model(xt,ts),data,ts)
        assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
        scaler.step(opt);scaler.update();torch.cuda.synchronize();times.append(time.perf_counter()-start);losses.append(loss.item())
    row={'parameters':sum(p.numel() for p in model.parameters()),'losses':losses,
         'batch32_median_ms':1000*float(np.median(times[4:])),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
    model.eval().requires_grad_(False);sam=FlowSampler(model,batch=4,steps=50,graph=True,amp=False);a=sam.run(991)
    del sam;gc.collect();torch.cuda.empty_cache()
    sam=FlowSampler(model,batch=4,steps=50,graph=False,amp=False);b=sam.run(991)
    row['trained_fp32_graph_eager_max_abs']=(a-b).abs().max().item();assert row['trained_fp32_graph_eager_max_abs']<2e-5
    row['finite_generation']=bool(torch.isfinite(a).all());assert row['finite_generation'] and a.shape==(4,channels,64,64);result['models'][kind]=row
    print(json.dumps({'kind':kind,**row}),flush=True)
    del sam,model,opt;gc.collect();torch.cuda.empty_cache()
Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
