import argparse, gc, json, time
from pathlib import Path
import numpy as np
import torch
from model import new_model, alpha_bar, gpu_status

p=argparse.ArgumentParser(); p.add_argument('--batch',type=int,default=32); p.add_argument('--fp32',action='store_true')
p.add_argument('--out',default='results/profile.json'); args=p.parse_args()
torch.set_num_threads(4); torch.manual_seed(42); torch.backends.cudnn.benchmark=True
before=gpu_status(); torch.cuda.reset_peak_memory_stats()
m=new_model().cuda().to(memory_format=torch.channels_last).train()
opt=torch.optim.AdamW(m.parameters(),lr=2e-4,fused=True); scaler=torch.amp.GradScaler('cuda',enabled=not args.fp32)
x=torch.randn(args.batch,3,64,64,device='cuda').to(memory_format=torch.channels_last)
t=torch.randint(0,1000,(args.batch,),device='cuda'); noise=torch.randn_like(x)
values=[]
for i in range(45):
    torch.cuda.synchronize(); start=time.perf_counter()
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16,enabled=not args.fp32):
        loss=torch.nn.functional.mse_loss(m(x,t,return_dict=False)[0],noise)
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    if i>=10: values.append((time.perf_counter()-start)*1000)
info={'batch':args.batch,'amp':not args.fp32,'parameters':sum(p.numel() for p in m.parameters()),
      'median_ms':float(np.median(values)),'p10_ms':float(np.percentile(values,10)),
      'p90_ms':float(np.percentile(values,90)),'all_ms':values,'images_per_second':args.batch*1000/float(np.median(values)),
      'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
      'before':before,'after':gpu_status(),'torch':torch.__version__,
      'note':'Synthetic training steps; no data transfer, EMA, validation or checkpoint. 10 warmups + 35 timed steps.'}
Path(args.out).write_text(json.dumps(info,indent=2)); print(json.dumps(info),flush=True)
