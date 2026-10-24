"""Small DDPG, batched tensor algebra with optional full-graph compilation.

The JS original performs SERIAL critic SGD. This implementation explicitly
uses a mean minibatch gradient. The parity test also implements the serial rule.
No Gym dependency; physics, reward, reset and uniform noise match the source.
"""
import argparse, copy, json, math, platform, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
KEYS = ['actor', 'critic', 'target_actor', 'target_critic']

def weights(inputs, device, dtype=torch.float32):
    result=[]
    for i, o in [(inputs,32),(32,16),(16,1)]:
        limit=math.sqrt((1 if o==1 else 2)/i)
        result.extend([torch.empty(o,i,device=device,dtype=dtype).uniform_(-limit,limit),torch.empty(o,device=device,dtype=dtype).uniform_(-0.005,0.005)])
    return result

def forward(w,x,actor=False):
    z1=x@w[0].T+w[1]; h1=z1.relu()
    z2=h1@w[2].T+w[3]; h2=z2.relu()
    z3=h2@w[4].T+w[5]
    y=z3.tanh()*10 if actor else z3
    return y,(x,z1,h1,z2,h2,z3)

def backward(w,c,dy,actor=False):
    x,z1,h1,z2,h2,z3=c
    g=dy*(10*(1-z3.tanh().square())) if actor else dy
    dw3=g.T@h2; db3=g.sum(0)
    g2=(g@w[4])*(z2>0)
    dw2=g2.T@h1; db2=g2.sum(0)
    g1=(g2@w[2])*(z1>0)
    return [g1.T@x,g1.sum(0),dw2,db2,dw3,db3],g1@w[0]

def physics(s,a):
    """Explicit Euler (using OLD velocities), post-step reward and termination."""
    x,xd,t,td=s.unbind(1)
    f=torch.nan_to_num(a[:,0],nan=0,posinf=0,neginf=0).clamp(-10,10)
    c=t.cos(); si=t.sin()
    temp=(f+0.05*td.square()*si)/1.1
    ta=(9.8*si-c*temp)/(0.5*(4/3-0.1*c.square()/1.1))
    xa=temp-0.05*ta*c/1.1
    ns=torch.stack([x+0.02*xd,xd+0.02*xa,t+0.02*td,td+0.02*ta],1)
    invalid=~torch.isfinite(ns).all(1) | ~torch.isfinite(s).all(1)
    ns=torch.where(invalid[:,None],torch.zeros_like(ns),ns)
    done=(ns[:,0].abs()>2.4)|(ns[:,2].abs()>math.pi/15)|invalid
    reward=torch.where(done,-10.,1.-0.5*ns[:,2].abs()/(math.pi/15)-0.1*ns[:,0].abs()/2.4)
    reward=torch.where(invalid,-100.,reward)
    return ns,reward[:,None],done[:,None]

class Agent:
    def __init__(self,device='cpu',dtype=torch.float32,optimizer='adam',lr_actor=1e-4,lr_critic=1e-3):
        self.device=device; self.dtype=dtype; self.optimizer=optimizer
        self.lr_actor=lr_actor; self.lr_critic=lr_critic
        self.actor=weights(4,device,dtype); self.critic=weights(5,device,dtype)
        self.target_actor=[w.clone() for w in self.actor];self.target_critic=[w.clone() for w in self.critic]
        self.m=[torch.zeros_like(w) for w in self.critic+self.actor]
        self.v=[torch.zeros_like(w) for w in self.critic+self.actor]
        self.t=torch.zeros((),device=device,dtype=dtype)
    def apply(self,w,grads,lr,offset):
        for i,(p,g) in enumerate(zip(w,grads)):
            if self.optimizer=='sgd': p.add_(g,alpha=-lr)
            else:
                m=self.m[offset+i]; v=self.v[offset+i]
                m.lerp_(g,0.1);v.mul_(0.999).addcmul_(g,g,value=0.001)
                p.addcdiv_(m/(1-0.9**self.t),(v/(1-0.999**self.t)).sqrt()+1e-8,value=-lr)
    def update(self,batch):
        s,a,r,ns,d=batch
        na,_=forward(self.target_actor,ns,True)
        nq,_=forward(self.target_critic,torch.cat((ns,na),1))
        y=r+0.99*(1-d)*nq
        q,cache=forward(self.critic,torch.cat((s,a),1))
        gc,_=backward(self.critic,cache,(q-y)/s.shape[0])
        self.t.add_(1);self.apply(self.critic,gc,self.lr_critic,0)
        aa,ca=forward(self.actor,s,True)
        aq,cq=forward(self.critic,torch.cat((s,aa),1))
        _,dx=backward(self.critic,cq,-torch.ones_like(aq)/s.shape[0])
        ga,_=backward(self.actor,ca,dx[:,-1:],True)
        self.apply(self.actor,ga,self.lr_actor,6)
        for target,online in [(self.target_actor,self.actor),(self.target_critic,self.critic)]:
            for tp,p in zip(target,online):tp.lerp_(p,0.005)
        return torch.stack([0.5*(q-y).square().mean(),-aq.mean()])
    def serial_update(self,batch):
        assert self.optimizer=='sgd'
        s,a,r,ns,d=batch
        na,_=forward(self.target_actor,ns,True);nq,_=forward(self.target_critic,torch.cat((ns,na),1))
        y=r+0.99*(1-d)*nq
        losses=[]
        for i in range(len(s)):
            q,c=forward(self.critic,torch.cat((s[i:i+1],a[i:i+1]),1));g,_=backward(self.critic,c,q-y[i:i+1]);losses.append(0.5*(q-y[i:i+1]).square().mean());self.apply(self.critic,g,self.lr_critic,0)
        aa,ca=forward(self.actor,s,True);aq,cq=forward(self.critic,torch.cat((s,aa),1));_,dx=backward(self.critic,cq,-torch.ones_like(aq)/len(s));ga,_=backward(self.actor,ca,dx[:,-1:],True);self.apply(self.actor,ga,self.lr_actor,6)
        for target,online in [(self.target_actor,self.actor),(self.target_critic,self.critic)]:
            for tp,p in zip(target,online):tp.lerp_(p,0.005)
        return torch.stack([torch.stack(losses).mean(),-aq.mean()])
    def export(self):return {k:[x.cpu().tolist() for x in getattr(self,k)] for k in KEYS}

class Runner:
    def __init__(self,agent,n=1,batch_size=64,capacity=50000,update=None):
        self.agent=agent;self.n=n;self.batch_size=batch_size;self.capacity=capacity
        d=agent.device;kw=dict(device=d,dtype=agent.dtype)
        self.state=torch.zeros(n,4,**kw);self.state[:,2:]=torch.rand(n,2,**kw)*0.02-0.01
        self.age=torch.zeros(n,1,device=d,dtype=torch.long)
        self.replay=torch.zeros(capacity,11,**kw)
        self.ptr=torch.zeros((),device=d,dtype=torch.long)
        self.size=torch.zeros((),device=d,dtype=torch.long)
        self.offset=torch.arange(n,device=d)
        self.update=update or agent.update
        self.last_loss=torch.zeros(2,**kw)
    def collect(self,random=False):
        a=forward(self.agent.actor,self.state,True)[0]
        a=torch.rand_like(a)*20-10 if random else (a+(torch.rand_like(a)*2-1)*2).clamp(-10,10)
        ns,r,done=physics(self.state,a)
        packed=torch.cat((self.state,a,r,ns,done.to(self.agent.dtype)),1)
        self.replay.index_copy_(0,(self.ptr+self.offset)%self.capacity,packed)
        self.ptr.copy_((self.ptr+self.n)%self.capacity)
        self.size.copy_((self.size+self.n).clamp(max=self.capacity))
        self.age.add_(1)
        reset=done|(self.age>=200)
        rs=torch.cat((torch.zeros_like(ns[:,:2]),torch.rand_like(ns[:,2:])*0.02-0.01),1)
        self.state.copy_(torch.where(reset,rs,ns));self.age.copy_(torch.where(reset,0,self.age))
    def sample(self):
        # Float -> index supports a device-resident dynamic size during graph replay.
        idx=(torch.rand(self.batch_size,device=self.agent.device)*self.size).long()
        b=self.replay[idx]
        return (b[:,:4],b[:,4:5],b[:,5:6],b[:,6:10],b[:,10:11])
    def step(self):
        self.collect();self.last_loss.copy_(self.update(self.sample()))


def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()

def capture(fn):
    """Warm up on a side stream, then capture; caller controls state restoration."""
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):fn()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fn()
    return graph

def tensors(obj):
    if torch.is_tensor(obj):return [obj]
    if isinstance(obj,(list,tuple)):return sum((tensors(x) for x in obj),[])
    if hasattr(obj,'__dict__'):return sum((tensors(v) for v in vars(obj).values() if not callable(v)),[])
    return []

def compiled_update(a,enabled):
    if not enabled:return a.update
    # Explicit CUDA graph below owns allocation lifetime and the entire loop.
    return torch.compile(a.update,fullgraph=True,mode='max-autotune-no-cudagraphs')

@torch.no_grad()
def evaluate(agent,episodes=64,horizon=200,trajectory=True):
    # Identical fixed evaluation states in JS and Python; independent of RNG.
    s=torch.zeros(episodes,4,device=agent.device,dtype=agent.dtype)
    i=torch.arange(episodes,device=agent.device,dtype=agent.dtype)
    s[:,2]=((i*37)%episodes+0.5)/episodes*0.02-0.01
    s[:,3]=((i*17)%episodes+0.5)/episodes*0.02-0.01
    alive=torch.ones(episodes,1,device=agent.device,dtype=torch.bool)
    lengths=torch.zeros(episodes,1,device=agent.device);returns=torch.zeros_like(lengths);traj=[]
    for i in range(horizon):
        a=forward(agent.actor,s,True)[0];ns,r,d=physics(s,a)
        if trajectory:traj.append(torch.cat((s[0],a[0],alive[0].to(agent.dtype))))
        lengths+=alive;returns+=r*alive;alive&=~d
        s=torch.where(alive,ns,s)
    result={'mean_length':lengths.mean().item(),'success200':(lengths>=200).float().mean().item(),'mean_return':returns.mean().item(),'horizon':horizon}
    if trajectory:
        rows=torch.stack(traj).cpu().tolist();result['trajectory']=[{'s':v[:4],'a':v[4]} for v in rows if v[5]]
    return result

@torch.no_grad()
def train(args):
    torch.manual_seed(args.seed)
    a=Agent(args.device,optimizer=args.optimizer,lr_actor=args.lr_actor,lr_critic=args.lr_critic)
    setup_start=time.perf_counter();update=compiled_update(a,args.compile and not args.compile_full)
    runner=Runner(a,args.envs,args.batch,update=update)
    # The warm-start is an explicit algorithm change from the original.
    for _ in range(math.ceil(args.warmup/args.envs)):runner.collect(random=True)
    step_fn=torch.compile(runner.step,fullgraph=True,mode='max-autotune-no-cudagraphs') if args.compile_full else runner.step
    if args.compile or args.compile_full:
        saved_compile=[(t,t.clone()) for t in tensors(runner)]
        step_fn()
        for t,v in saved_compile:t.copy_(v)
    graph=None
    if args.graph:
        saved=[(t,t.clone()) for t in tensors(runner)]
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
        graph=capture(step_fn)
        for t,v in saved:t.copy_(v)
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
    sync(args.device);setup_seconds=time.perf_counter()-setup_start
    history=[];train_seconds=0.;count=0
    for stop in range(args.eval_every,args.updates+args.eval_every,args.eval_every):
        stop=min(stop,args.updates);sync(args.device);start=time.perf_counter()
        for _ in range(stop-count):
            if graph:graph.replay()
            else:step_fn()
        sync(args.device);train_seconds+=time.perf_counter()-start;count=stop
        ev=evaluate(a,trajectory=False)
        entry={'updates':count,'transitions':count*args.envs+args.warmup,'train_seconds':train_seconds,'wall_seconds':time.perf_counter()-setup_start,'loss':runner.last_loss.cpu().tolist(),**ev}
        history.append(entry);print(json.dumps(entry),flush=True)
        if not all(math.isfinite(x) for x in entry['loss']):raise RuntimeError('Nonfinite training loss')
    result={'kind':'tensor_train','config':vars(args),'setup_seconds':setup_seconds,'train_seconds':train_seconds,'wall_seconds':time.perf_counter()-setup_start,'history':history,'evaluation':evaluate(a),'long_evaluation':evaluate(a,horizon=1000,trajectory=False),'actor':a.export()['actor'],'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else None}
    result['wall_seconds']=time.perf_counter()-setup_start
    Path(args.out).write_text(json.dumps(result,indent=2));return result

@torch.no_grad()
def benchmark(args):
    rows=[]
    for device,compile_,graph_,envs,batch in [('cpu',False,False,1,64),('cuda',False,False,1,64),('cuda',False,True,1,64),('cuda',True,True,1,64),('cuda',True,True,32,256),('cuda',True,True,256,1024)]:
        torch.manual_seed(1);a=Agent(device,optimizer='adam');update=compiled_update(a,compile_);runner=Runner(a,envs,batch,update=update)
        for _ in range(math.ceil(5000/envs)):runner.collect(random=True)
        sync(device);start=time.perf_counter()
        for _ in range(5):runner.step()
        graph=capture(runner.step) if graph_ else None
        sync(device);setup=time.perf_counter()-start
        rounds=[]
        for _ in range(5):
            sync(device);start=time.perf_counter()
            for i in range(args.updates):
                if graph:graph.replay()
                else:runner.step()
            sync(device);rounds.append((time.perf_counter()-start)*1000/args.updates)
        median=sorted(rounds)[2]
        row={'device':device,'compile':compile_,'graph':graph_,'envs':envs,'batch':batch,'rounds_ms':rounds,'median_ms':median,'transitions_per_second':envs*1000/median,'updates_per_second':1000/median,'setup_seconds':setup,'scope':'full collect + replay write/sample + minibatch Adam update + Polyak','optimizer':'adam','finite':bool(torch.isfinite(runner.last_loss).all().item())}
        rows.append(row);print(json.dumps(row),flush=True)
        Path(args.out).write_text(json.dumps({'hardware':hardware(),'rows':rows},indent=2))

def hardware():
    return {'host':platform.node(),'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,'cpu_threads':torch.get_num_threads()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','bench']);p.add_argument('--device',default='cuda');p.add_argument('--envs',type=int,default=32);p.add_argument('--batch',type=int,default=256);p.add_argument('--updates',type=int,default=20000);p.add_argument('--seed',type=int,default=1);p.add_argument('--warmup',type=int,default=1024);p.add_argument('--eval-every',type=int,default=2000);p.add_argument('--optimizer',choices=['adam','sgd'],default='adam');p.add_argument('--lr-actor',type=float,default=1e-4);p.add_argument('--lr-critic',type=float,default=1e-3);p.add_argument('--compile',action='store_true');p.add_argument('--graph',action='store_true');p.add_argument('--compile-full',action='store_true');p.add_argument('--out',default='results/train.json');args=p.parse_args()
    if args.envs<1 or args.envs>50000 or args.batch<1 or args.updates<1: p.error('Invalid shape/count')
    if args.graph and args.device!='cuda':p.error('--graph requires --device cuda')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.mode=='train':train(args)
    else:benchmark(args)
