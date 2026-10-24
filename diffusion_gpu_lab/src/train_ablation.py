"""Interleaved training timings plus paired AMP gradient validation."""
import copy,json,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from fast import new_model,schedule
from core import q_sample
from benchmark import summary,status
from train_graph import GraphTrainer

def main():
    torch.set_num_threads(4);torch.manual_seed(19);torch.backends.cudnn.benchmark=True
    out=Path('results');m=new_model().cuda();m.load_state_dict(torch.load(out/'checkpoint_epoch20.pt',weights_only=True)['state_dict'])
    cpu=torch.from_numpy(np.load('data/mnist_subset.npz')['train_x'][:64]).unsqueeze(1).float()/127.5-1
    resident=cpu.cuda();p=schedule();t=torch.randint(0,200,(64,),device='cuda');noise=torch.randn_like(resident)
    xt=q_sample(resident,t,p,noise);grads=[];losses=[]
    for amp in [False,True]:
        model=copy.deepcopy(m);model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):loss=F.mse_loss(model(xt,t.float()),noise)
        loss.backward();grads.append(torch.cat([v.grad.flatten() for v in model.parameters()]));losses.append(loss.item())
    gradient_relative_l2=((grads[1]-grads[0]).norm()/grads[0].norm()).item()
    assert gradient_relative_l2<.1 and abs(losses[1]-losses[0])/losses[0]<.02
    (out/'gradient_validation.json').write_text(json.dumps({'fp32_loss':losses[0],'amp_loss':losses[1],
        'gradient_relative_l2':gradient_relative_l2,'gradient_threshold':.1,'loss_relative_threshold':.02,'passed':True},indent=2))
    states={}
    for name,amp,cl,fused in [('original_fp32',False,False,False),('gpu_resident_fp32',False,True,True),('amp_channels_last',True,True,True)]:
        model=copy.deepcopy(m)
        if cl:model.to(memory_format=torch.channels_last)
        states[name]={'model':model,'amp':amp,'cl':cl,'opt':torch.optim.Adam(model.parameters(),lr=1e-4,fused=fused),
                      'scaler':torch.amp.GradScaler('cuda',enabled=amp),'x':resident.contiguous(memory_format=torch.channels_last if cl else torch.contiguous_format)}
    states['graph_fp32']={'graph':GraphTrainer(copy.deepcopy(m)), 'x':resident}
    def step(s):
        if 'graph' in s:return s['graph'].step(s['x'])
        s['opt'].zero_grad(set_to_none=True);x=s['x'] if s['cl'] else cpu.cuda()
        t=torch.randint(0,200,(64,),device='cuda');n=torch.randn_like(x);xt=q_sample(x,t,p,n)
        with torch.autocast('cuda',dtype=torch.float16,enabled=s['amp']):loss=F.mse_loss(s['model'](xt,t.float()),n)
        if not s['cl']:loss.item()
        s['scaler'].scale(loss).backward();s['scaler'].step(s['opt']);s['scaler'].update()
        return loss
    for s in states.values():
        for _ in range(8):step(s)
    values={k:[] for k in states};telemetry=[];rng=random.Random(72)
    for rep in range(7):
        names=list(states);rng.shuffle(names);telemetry.append({'rep':rep,'order':names,'gpu':status()})
        for name in names:
            torch.cuda.synchronize();started=time.perf_counter()
            for _ in range(20):loss=step(states[name])
            torch.cuda.synchronize();ms=(time.perf_counter()-started)*1000/20
            assert torch.isfinite(loss).item();values[name].append(ms);print(rep,name,ms,flush=True)
    rows=[{'name':name,**summary(v),'batch':64,
           'setup_seconds':states[name]['graph'].setup_seconds if 'graph' in states[name] else None}
          for name,v in values.items()]
    (out/'training_benchmark.json').write_text(json.dumps(rows,indent=2))
    (out/'training_benchmark_metadata.json').write_text(json.dumps({'scope':'20 full GPU updates per timed block, 7 interleaved blocks per mode, same 64 real images; original includes H2D and loss.item; excludes PIL/DataLoader/HTTP',
            'telemetry':telemetry,'warmup_steps':8},indent=2))

if __name__=='__main__':main()
