"""Faithful CNN port plus measured CPU/eager/CUDA Graph implementations.

The default architecture matches original/train.html. PyTorch uses fused
log-softmax cross entropy and integer labels (same classification objective).
NHWC flatten ordering and Glorot initialization match the TF.js convention.
"""
import argparse, copy, gc, json, math, os, platform, statistics, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self,filters=(8,16,32),dense=64,kernel=3,pool=2):
        super().__init__()
        if not filters or len(filters)>3 or any(c<1 for c in filters) or kernel%2!=1 or pool<1:
            raise ValueError('Use 1–3 positive channel counts, odd kernel and positive pool')
        self.convs=nn.ModuleList(); n=1; h=28; self.pool=pool
        for c in filters:
            self.convs.append(nn.Conv2d(n,c,kernel,padding=kernel//2)); n=c; h//=pool
        if h<1: raise ValueError('Pooling collapses spatial dimensions')
        self.fc=nn.Linear(h*h*n,dense); self.out=nn.Linear(dense,10)
        for layer in [*self.convs,self.fc,self.out]:
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self,x):
        for conv in self.convs: x=F.max_pool2d(F.relu(conv(x)),self.pool)
        x=x.permute(0,2,3,1).flatten(1)  # preserve TensorFlow NHWC flatten order
        return self.out(F.relu(self.fc(x)))

def synchronize(device):
    if device=='cuda': torch.cuda.synchronize()

def make_data(path,device,train_n=60000,test_n=10000):
    d=np.load(path)
    return (torch.from_numpy(d['train_x'][:train_n,None].copy()).to(device).float().div_(255),
            torch.from_numpy(d['train_y'][:train_n].copy()).to(device),
            torch.from_numpy(d['test_x'][:test_n,None].copy()).to(device).float().div_(255),
            torch.from_numpy(d['test_y'][:test_n].copy()).to(device))

class Runner:
    def __init__(self,model,x,y,mode,batch,lr):
        self.model,self.x,self.y,self.mode,self.batch=model,x,y,mode,batch
        self.device=x.device.type; self.amp='amp' in mode; self.graph='graph' in mode
        self.channels_last=('cl' in mode or self.amp)
        if self.channels_last: self.model.to(memory_format=torch.channels_last)
        self.opt=torch.optim.Adam(model.parameters(),lr=lr,eps=1e-7,
                                  fused=self.device=='cuda' and mode!='cuda_eager',
                                  capturable=self.graph)
        self.ids=torch.arange(batch,device=x.device)
        self.metrics=torch.zeros(3,device=x.device)
        self.loss_scale=128.0 if self.amp else 1.0
        self.setup_seconds=0.
        if self.graph:
            t=time.perf_counter(); initial=copy.deepcopy(model.state_dict())
            stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(12): self._step(self.ids)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g): self._step(self.ids)
            # Warmup/capture must not train on extra data. Restore parameters and Adam state in place.
            self.model.load_state_dict(initial)
            for state in self.opt.state.values():
                for value in state.values():
                    if torch.is_tensor(value): value.zero_()
            self.metrics.zero_(); torch.cuda.synchronize()
            self.setup_seconds=time.perf_counter()-t

    def _step(self,ids):
        # Graph replay and the smaller eager tail share persistent gradient buffers.
        # Rebinding .grad for the tail would break the captured gradient addresses.
        self.opt.zero_grad(set_to_none=not self.graph)
        x=self.x[ids]; y=self.y[ids]
        if self.channels_last: x=x.contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.amp):
            logits=self.model(x); loss=F.cross_entropy(logits,y)
        (loss*self.loss_scale).backward()
        if self.amp:
            for p in self.model.parameters(): p.grad.mul_(1/self.loss_scale)
        self.opt.step()
        with torch.no_grad():
            self.metrics[0].add_(loss.detach()*len(ids))
            self.metrics[1].add_((logits.argmax(1)==y).sum())
            self.metrics[2].add_(len(ids))

    def step(self,ids):
        if self.graph and len(ids)==self.batch:
            self.ids.copy_(ids); self.g.replay()
        else:
            # Exact remainder, no dropped examples or repeated padding samples.
            self._step(ids)

@torch.no_grad()
def evaluate(model,x,y,amp=False,channels_last=False,return_probs=False):
    correct=torch.zeros((),device=x.device,dtype=torch.int64); loss=torch.zeros((),device=x.device); probs=[]
    for start in range(0,len(x),1024):
        xb=x[start:start+1024]; yb=y[start:start+1024]
        if channels_last: xb=xb.contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            logits=model(xb); loss+=F.cross_entropy(logits,yb,reduction='sum')
        correct+=(logits.argmax(1)==yb).sum()
        if return_probs: probs.append(logits.float().softmax(1).cpu().numpy())
    ans={'accuracy':correct.item()/len(x),'loss':loss.item()/len(x)}
    if return_probs: ans['probabilities']=np.concatenate(probs)
    return ans

def environment():
    return {'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,
            'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(),
            'cpu':subprocess.check_output(['bash','-c',"lscpu | sed -n 's/^Model name: *//p'"],text=True).strip(),
            'cpu_threads':torch.get_num_threads(),
            'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),
            'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}

def model_for(args,device):
    torch.manual_seed(args.seed)
    return CNN(args.filters,args.dense,args.kernel,args.pool).to(device)

def benchmark(args):
    out={'environment':environment(),'config':vars(args),'results':[]}
    for mode in args.modes.split(','):
        dev='cpu' if mode=='cpu' else 'cuda'
        x,y,_,_=make_data(Path(args.data)/'mnist.npz',dev,args.train_n,args.test_n)
        m=model_for(args,dev); r=Runner(m,x,y,mode,args.batch,args.lr)
        ids=torch.arange(args.batch,device=dev)
        for _ in range(20): r.step(ids)
        synchronize(dev)
        samples=[]
        for _ in range(args.repeats):
            t=time.perf_counter()
            for _ in range(args.steps): r.step(ids)
            synchronize(dev); samples.append((time.perf_counter()-t)*1000/args.steps)
        result={'mode':mode,'step_ms':samples,'median_ms':statistics.median(samples),
                'images_per_second':1000*args.batch/statistics.median(samples),'graph_setup_s':r.setup_seconds,
                'parameter_count':sum(p.numel() for p in m.parameters())}
        out['results'].append(result); print(json.dumps(result),flush=True)
        Path(args.output).write_text(json.dumps(out,indent=2))
        del r,m,x,y; gc.collect(); torch.cuda.empty_cache()

def correctness(args):
    torch.backends.cudnn.benchmark=False
    x,y,_,_=make_data(Path(args.data)/'mnist.npz','cuda',6000,1000)
    m=model_for(args,'cuda'); state=copy.deepcopy(m.state_dict())
    ref=Runner(m,x,y,'cuda_fused',args.batch,args.lr)
    graph_m=model_for(args,'cuda'); graph=Runner(graph_m,x,y,'cuda_graph',args.batch,args.lr)
    init_err=max((state[k]-graph_m.state_dict()[k]).abs().max().item() for k in state)
    cpu=copy.deepcopy(m).cpu()
    with torch.no_grad(): forward_error=(cpu(x[:32].cpu()).cuda()-m(x[:32])).abs().max().item()
    # Different batches, including an eager tail followed by another graph replay.
    for i,size in enumerate([args.batch,args.batch,17,args.batch,args.batch]):
        ids=torch.arange(i*args.batch,i*args.batch+size,device='cuda')
        ref.step(ids); graph.step(ids)
    synchronize('cuda')
    weight_error=max((m.state_dict()[k]-graph_m.state_dict()[k]).abs().max().item() for k in state)
    metrics_error=(ref.metrics-graph.metrics).abs().max().item()
    adam_error=max((ref.opt.state[p][key]-graph.opt.state[q][key]).abs().max().item()
                   for p,q in zip(m.parameters(),graph_m.parameters()) for key in ('exp_avg','exp_avg_sq','step'))
    result={'warmup_restore_max_abs':init_err,'cpu_gpu_logits_max_abs':forward_error,
            'eager_graph_weights_max_abs':weight_error,'eager_graph_adam_max_abs':adam_error,
            'eager_graph_metrics_max_abs':metrics_error,'steps':5,'includes_tail_and_replay':True}
    result['passed']=init_err==0 and forward_error<1e-4 and weight_error<1e-5 and adam_error<1e-5
    Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
    assert result['passed'],result

def train(args):
    outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    dev='cpu' if args.mode=='cpu' else 'cuda'
    t0=time.perf_counter(); x,y,tx,ty=make_data(Path(args.data)/'mnist.npz',dev,args.train_n,args.test_n)
    synchronize(dev); load_s=time.perf_counter()-t0
    model=model_for(args,dev); r=Runner(model,x,y,args.mode,args.batch,args.lr)
    if dev=='cuda': torch.cuda.reset_peak_memory_stats()
    out={'environment':environment(),'config':vars(args),'load_seconds':load_s,
         'graph_setup_seconds':r.setup_seconds,'parameter_count':sum(p.numel() for p in model.parameters()),'epochs':[]}
    # One shared CPU permutation per epoch makes CPU/GPU sample orders identical.
    # The tiny index array is copied once; images and labels remain on the device.
    gen=torch.Generator().manual_seed(args.seed+1000)
    run_start=time.perf_counter()
    for epoch in range(args.epochs):
        r.metrics.zero_(); synchronize(dev); start=time.perf_counter()
        order=torch.randperm(len(x),generator=gen).to(dev)
        for ids in order.split(args.batch): r.step(ids)
        synchronize(dev); train_s=time.perf_counter()-start
        loss_sum,correct,n=r.metrics.cpu().tolist()
        if not math.isfinite(loss_sum) or any(not torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError('Non-finite training; reduce loss scale/lr or use FP32')
        e0=time.perf_counter()
        val=evaluate(model,tx,ty,r.amp,r.channels_last); synchronize(dev)
        eval_s=time.perf_counter()-e0
        row={'epoch':epoch+1,'train_accuracy':correct/n,'train_loss':loss_sum/n,'test_accuracy':val['accuracy'],
             'test_loss':val['loss'],'train_seconds':train_s,'test_seconds':eval_s,
             'samples':int(n),'cumulative_seconds':time.perf_counter()-run_start}
        out['epochs'].append(row); print(json.dumps(row),flush=True)
        (outdir/'metrics.json').write_text(json.dumps(out,indent=2))
    out['run_seconds']=time.perf_counter()-run_start
    out['train_seconds']=sum(e['train_seconds'] for e in out['epochs'])
    out['train_and_test_seconds']=sum(e['train_seconds']+e['test_seconds'] for e in out['epochs'])
    out['pipeline_seconds_before_export']=time.perf_counter()-t0
    out['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20 if dev=='cuda' else None
    out['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20 if dev=='cuda' else None
    final=evaluate(model,tx,ty,r.amp,r.channels_last,True)
    probs=final.pop('probabilities'); np.save(outdir/'test_probabilities.npy',probs)
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save({'state_dict':state,'config':vars(args),'metrics':out},outdir/'model.pt')
    portable={'architecture':{'filters':args.filters,'dense':args.dense,'kernel':args.kernel,'pool':args.pool,
                             'flatten':'NHWC','normalization':'uint8 / 255'},
              'tensors':{k:{'shape':list(v.shape),'data':v.flatten().tolist()} for k,v in state.items()}}
    (outdir/'model.json').write_text(json.dumps(portable,separators=(',',':')))
    # Read exported checkpoint back and verify every test prediction, not just file existence.
    loaded=torch.load(outdir/'model.pt',map_location=dev,weights_only=False)
    restored=model_for(args,dev); restored.load_state_dict(loaded['state_dict'])
    if r.channels_last: restored.to(memory_format=torch.channels_last)
    again=evaluate(restored,tx,ty,r.amp,r.channels_last,True)['probabilities']
    out['export_max_probability_error']=float(np.max(np.abs(probs-again)))
    out['export_prediction_agreement']=float(np.mean(probs.argmax(1)==again.argmax(1)))
    (outdir/'metrics.json').write_text(json.dumps(out,indent=2))
    print('DONE',outdir,'accuracy',final['accuracy'],'seconds',out['train_and_test_seconds'],flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['bench','check','train'])
    p.add_argument('--data',default='data'); p.add_argument('--output',default='results/run')
    p.add_argument('--mode',default='cuda_graph'); p.add_argument('--modes',default='cpu,cuda_eager,cuda_fused,cuda_graph,cuda_graph_cl,cuda_graph_amp')
    p.add_argument('--filters',type=lambda s:[int(x) for x in s.split(',')],default=[8,16,32])
    p.add_argument('--dense',type=int,default=64); p.add_argument('--kernel',type=int,default=3); p.add_argument('--pool',type=int,default=2)
    p.add_argument('--batch',type=int,default=128); p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--epochs',type=int,default=5); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train-n',type=int,default=60000); p.add_argument('--test-n',type=int,default=10000)
    p.add_argument('--threads',type=int,default=4); p.add_argument('--steps',type=int,default=80); p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    if not 1<=args.batch<=args.train_n<=60000 or not 1<=args.test_n<=10000: p.error('Invalid batch/data size')
    if args.epochs<1 or args.dense<1 or args.lr<=0 or args.threads<1 or args.steps<1 or args.repeats<1:
        p.error('epochs, dense, lr, threads, steps and repeats must be positive')
    if args.mode not in ['cpu','cuda_eager','cuda_fused','cuda_graph','cuda_graph_cl','cuda_graph_amp']: p.error('Unknown mode')
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=True; torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    {'bench':benchmark,'check':correctness,'train':train}[args.action](args)

if __name__=='__main__': main()
