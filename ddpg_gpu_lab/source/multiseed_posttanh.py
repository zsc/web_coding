"""Independent DDPG runs packed along a leading seed dimension.

No parameter, replay or optimizer sharing between seeds. Stateless integer
random streams are keyed by seed, step, element and stream, so batching seeds
in a different order does not couple their exploratory data.
"""
import argparse,copy,json,math,time
from pathlib import Path
import torch
from ddpg import Agent,physics,capture,hardware

def fw(w,x,actor=False):
    z1=torch.bmm(x,w[0].transpose(1,2))+w[1];h1=z1.relu()
    z2=torch.bmm(h1,w[2].transpose(1,2))+w[3];h2=z2.relu()
    z3=torch.bmm(h2,w[4].transpose(1,2))+w[5]
    y=z3.tanh()*10 if actor else z3
    return y,(x,z1,h1,z2,h2,z3)

def bw(w,c,dy,actor=False):
    x,z1,h1,z2,h2,z3=c
    g=dy*(10*(1-z3.tanh().square())) if actor else dy
    dw3=torch.bmm(g.transpose(1,2),h2);db3=g.sum(1,keepdim=True)
    g2=torch.bmm(g,w[4])*(z2>0)
    dw2=torch.bmm(g2.transpose(1,2),h1);db2=g2.sum(1,keepdim=True)
    g1=torch.bmm(g2,w[2])*(z1>0)
    return [torch.bmm(g1.transpose(1,2),x),g1.sum(1,keepdim=True),dw2,db2,dw3,db3],torch.bmm(g1,w[0])

def pack(items,key):
    return [torch.stack([getattr(a,key)[j] for a in items]).unsqueeze(1) if j%2 else torch.stack([getattr(a,key)[j] for a in items]) for j in range(6)]

def uniform(seed,tick,idx,stream):
    # Integer low-bias hash; all multiplies stay within signed int64 range.
    x=(seed*747796405 + tick*65537 + idx*(-1403630843) + stream*277803737) & 0xffffffff
    x=((x^(x>>16))*0x45d9f3b)&0xffffffff
    x=((x^(x>>16))*0x45d9f3b)&0xffffffff
    x=x^(x>>16)
    return (x>>8).to(torch.float32)*(1./16777216.)

class Packed:
    def __init__(self,seeds,device='cuda',dtype=torch.float32,lr_actor=1e-4,lr_critic=1e-3):
        self.device=device;self.dtype=dtype;self.seeds=list(seeds);self.k=len(seeds)
        items=[]
        for seed in seeds:
            torch.manual_seed(seed);items.append(Agent('cpu',dtype,lr_actor=lr_actor,lr_critic=lr_critic))
        for key in ['actor','critic','target_actor','target_critic']:setattr(self,key,[w.to(device) for w in pack(items,key)])
        self.m=[torch.zeros_like(w) for w in self.critic+self.actor];self.v=[w.clone() for w in self.m]
        self.t=torch.zeros((),device=device,dtype=dtype)
        self.lr_actor=lr_actor;self.lr_critic=lr_critic
        self.active=torch.ones(self.k,1,1,device=device,dtype=dtype)
    def action(self,s):return fw(self.actor,s,True)[0]
    def apply(self,w,gs,lr,offset):
        for j,(p,g) in enumerate(zip(w,gs)):
            m=self.m[offset+j];v=self.v[offset+j]
            m.lerp_(g,0.1);v.mul_(0.999).addcmul_(g,g,value=0.001)
            p.add_(-lr*self.active*(m/(1-0.9**self.t))/((v/(1-0.999**self.t)).sqrt()+1e-8))
    def update(self,b):
        s,a,r,ns,d=b;batch=s.shape[1]
        na,_=fw(self.target_actor,ns,True);nq,_=fw(self.target_critic,torch.cat((ns,na),2));y=r+.99*(1-d)*nq
        q,c=fw(self.critic,torch.cat((s,a),2));gc,_=bw(self.critic,c,(q-y)/batch)
        self.t.add_(1);self.apply(self.critic,gc,self.lr_critic,0)
        aa,ca=fw(self.actor,s,True);aq,cq=fw(self.critic,torch.cat((s,aa),2));_,dx=bw(self.critic,cq,-torch.ones_like(aq)/batch);ga,_=bw(self.actor,ca,dx[:,:,-1:],True);self.apply(self.actor,ga,self.lr_actor,6)
        for target,online in [(self.target_actor,self.actor),(self.target_critic,self.critic)]:
            for tp,p in zip(target,online):tp.add_((p-tp)*(.005*self.active))
        return torch.stack([.5*(q-y).square().mean((1,2)),-aq.mean((1,2))],1)

class PackedTD3(Packed):
    """TD3 stability variant: twin critics, delayed actor, smoothed targets.

    State/action scaling is explicit. Physics, rewards and reset distribution
    remain unchanged. A bounded per-seed gradient norm prevents outlier updates.
    """
    def __init__(self,*args,symmetry=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.symmetry=symmetry
        items=[]
        for seed in self.seeds:
            torch.manual_seed(seed+1000000);items.append(Agent('cpu',self.dtype))
        self.critic2=[w.to(self.device) for w in pack(items,'critic')]
        self.target_critic2=[w.clone() for w in self.critic2]
        self.m=[torch.zeros_like(w) for w in self.critic+self.critic2+self.actor]
        self.v=[w.clone() for w in self.m]
        self.obs_scale=torch.tensor([1/2.4,.5,1/(math.pi/15),.5],device=self.device,dtype=self.dtype).reshape(1,1,4)
        self.seedkey=torch.tensor(self.seeds,device=self.device,dtype=torch.long)[:,None]
    def policy(self,w,s):
        pos,cp=fw(w,s*self.obs_scale,True)
        if not self.symmetry:return pos,(cp,None)
        neg,cn=fw(w,-s*self.obs_scale,True)
        return (pos-neg)*.5,(cp,cn)
    def policy_backward(self,cache,dy):
        cp,cn=cache
        if cn is None:return bw(self.actor,cp,dy,True)[0]
        gp,_=bw(self.actor,cp,dy*.5,True);gn,_=bw(self.actor,cn,-dy*.5,True)
        return [p+n for p,n in zip(gp,gn)]
    def action(self,s):return self.policy(self.actor,s)[0]
    def cin(self,s,a):return torch.cat((s*self.obs_scale,a/10),2)
    def apply_td3(self,w,gs,lr,offset,actor=False):
        norm=torch.stack([g.square().sum((1,2),keepdim=True) for g in gs]).sum(0).sqrt()
        clip=(10./(norm+1e-8)).clamp(max=1.)
        gate=(self.t.remainder(2)==0).to(self.dtype) if actor else torch.ones_like(self.t)
        opt_t=(self.t/2).floor().clamp(min=1) if actor else self.t
        for j,(p,g) in enumerate(zip(w,gs)):
            g=g*clip;m=self.m[offset+j];v=self.v[offset+j]
            m.add_((g-m)*(.1*gate));v.add_((g.square()-v)*(.001*gate))
            p.add_(-lr*self.active*gate*(m/(1-.9**opt_t))/((v/(1-.999**opt_t)).sqrt()+1e-8))
    def update(self,b):
        s,a,r,ns,d=b
        if self.symmetry:
            s=torch.cat((s,-s),1);a=torch.cat((a,-a),1);ns=torch.cat((ns,-ns),1);r=torch.cat((r,r),1);d=torch.cat((d,d),1)
        batch=s.shape[1];idx=torch.arange(batch,device=self.device)[None,:]
        u1=uniform(self.seedkey,self.t.long(),idx,11).clamp(min=1e-7);u2=uniform(self.seedkey,self.t.long(),idx,12)
        smoothing=((-2*u1.log()).sqrt()*(2*math.pi*u2).cos()*2).clamp(-5,5).unsqueeze(-1)
        na,_=self.policy(self.target_actor,ns);na=(na+smoothing).clamp(-10,10)
        nq,_=fw(self.target_critic,self.cin(ns,na));nq2,_=fw(self.target_critic2,self.cin(ns,na));y=r+.99*(1-d)*torch.minimum(nq,nq2)
        q,c=fw(self.critic,self.cin(s,a));q2,c2=fw(self.critic2,self.cin(s,a))
        gc,_=bw(self.critic,c,(q-y)/batch);gc2,_=bw(self.critic2,c2,(q2-y)/batch)
        self.t.add_(1);self.apply_td3(self.critic,gc,self.lr_critic,0);self.apply_td3(self.critic2,gc2,self.lr_critic,6)
        aa,ca=self.policy(self.actor,s);aq,cq=fw(self.critic,self.cin(s,aa))
        _,dx=bw(self.critic,cq,-torch.ones_like(aq)/batch);ga=self.policy_backward(ca,dx[:,:,-1:]/10)
        self.apply_td3(self.actor,ga,self.lr_actor,12,True)
        gate=(self.t.remainder(2)==0).to(self.dtype)
        for target,online in [(self.target_actor,self.actor),(self.target_critic,self.critic),(self.target_critic2,self.critic2)]:
            for tp,p in zip(target,online):tp.add_((p-tp)*(.005*self.active*gate))
        return torch.stack([.25*((q-y).square()+(q2-y).square()).mean((1,2)),-aq.mean((1,2))],1)

class PackedRunner:
    def __init__(self,a,n=32,batch=256,capacity=50000,episode_limit=200):
        self.agent=a;self.n=n;self.batch=batch;self.capacity=capacity;self.episode_limit=episode_limit
        d=a.device;k=a.k;kw=dict(device=d,dtype=a.dtype)
        self.seedkey=torch.tensor(a.seeds,device=d,dtype=torch.long)[:,None]
        self.tick=torch.zeros((),device=d,dtype=torch.long)
        self.idx_noise=torch.arange(n,device=d)[None,:]
        self.idx_reset=torch.arange(n*2,device=d)[None,:]
        self.idx_sample=torch.arange(batch,device=d)[None,:]
        self.state=torch.zeros(k,n,4,**kw);self.state[:,:,2:]=(self.rand(self.idx_reset,3).reshape(k,n,2)*.02-.01).to(a.dtype)
        self.age=torch.zeros(k,n,1,device=d,dtype=torch.long)
        self.replay=torch.zeros(k,capacity,11,**kw)
        self.ptr=torch.zeros((),device=d,dtype=torch.long);self.size=torch.zeros_like(self.ptr)
        self.last_loss=torch.zeros(k,2,**kw)
    def rand(self,idx,stream):return uniform(self.seedkey,self.tick,idx,stream)
    def prepare(self,random=False):
        self.tick.add_(1)
        a=self.agent.action(self.state)
        noise=self.rand(self.idx_noise,1).unsqueeze(-1).to(self.agent.dtype)
        a=noise*20-10 if random else (a+(noise*2-1)*2).clamp(-10,10)
        ns,r,done=physics(self.state.reshape(-1,4),a.reshape(-1,1));ns=ns.reshape(self.agent.k,self.n,4);r=r.reshape(self.agent.k,self.n,1);done=done.reshape(self.agent.k,self.n,1)
        packed=torch.cat((self.state,a,r,ns,done.to(self.agent.dtype)),2)
        index=((self.ptr+self.idx_noise)%self.capacity).unsqueeze(-1).expand(self.agent.k,-1,11)
        self.ptr.copy_((self.ptr+self.n)%self.capacity);self.size.copy_((self.size+self.n).clamp(max=self.capacity))
        self.age.add_(1);reset=done|(self.age>=self.episode_limit)
        rs=torch.cat((torch.zeros_like(ns[:,:,:2]),(self.rand(self.idx_reset,3).reshape(self.agent.k,self.n,2)*.02-.01).to(self.agent.dtype)),2)
        self.state.copy_(torch.where(reset,rs,ns));self.age.copy_(torch.where(reset,0,self.age))
        return index,packed
    def collect(self,random=False):
        index,packed=self.prepare(random)
        self.replay.scatter_(1,index,packed)
    def sample(self):
        idx=(self.rand(self.idx_sample,2)*self.size).long().unsqueeze(-1).expand(-1,-1,11)
        b=torch.gather(self.replay,1,idx)
        return b[:,:,:4],b[:,:,4:5],b[:,:,5:6],b[:,:,6:10],b[:,:,10:11]
    def learn(self):
        self.last_loss.copy_(self.agent.update(self.sample()))
    def step(self):
        self.collect();self.learn()

class Evaluator:
    def __init__(self,a,episodes=64):
        self.a=a;self.episodes=episodes;kw=dict(device=a.device,dtype=a.dtype)
        self.state=torch.zeros(a.k,episodes,4,**kw)
        self.alive=torch.ones(a.k,episodes,1,device=a.device,dtype=torch.bool)
        self.lengths=torch.zeros(a.k,episodes,1,device=a.device,dtype=torch.long)
        self.max_angle=torch.zeros(a.k,episodes,1,**kw);self.max_pos=torch.zeros_like(self.max_angle)
        self.max_tail=torch.zeros(a.k,episodes,4,**kw)
        self.tail_gate=torch.zeros((),device=a.device,dtype=a.dtype)
    def reset(self,split=0):
        self.state.zero_();self.alive.fill_(True);self.lengths.zero_();self.max_angle.zero_();self.max_pos.zero_();self.max_tail.zero_();self.tail_gate.zero_()
        idx=torch.arange(self.episodes*2,device=self.a.device)[None,:]
        # Same validation initial states for each learner; held-out domains separate.
        u=uniform(torch.tensor([[20261024]],device=self.a.device),torch.tensor(split,device=self.a.device),idx,77).reshape(1,self.episodes,2)
        self.state[:,:,2:].copy_((u*.02-.01).expand(self.a.k,-1,-1))
    def step(self):
        a=self.a.action(self.state)
        ns,_,done=physics(self.state.reshape(-1,4),a.reshape(-1,1))
        ns=ns.reshape_as(self.state);done=done.reshape_as(self.alive)
        self.lengths.add_(self.alive.long());self.alive.logical_and_(~done)
        self.state.copy_(torch.where(self.alive,ns,self.state))
        self.max_angle.copy_(torch.maximum(self.max_angle,ns[:,:,2:3].abs()))
        self.max_pos.copy_(torch.maximum(self.max_pos,ns[:,:,:1].abs()))
        self.max_tail.copy_(torch.maximum(self.max_tail,ns.abs()*self.tail_gate))
    def result(self,horizon):
        lengths=self.lengths[:,:,0].cpu();alive=self.alive[:,:,0].cpu();tail=self.max_tail.cpu();rows=[]
        for j,seed in enumerate(self.a.seeds):
            rows.append({'seed':seed,'horizon':horizon,'episodes':self.episodes,'survived':int(alive[j].sum()),'success_rate':float(alive[j].float().mean()),'minimum_steps':int(lengths[j].min()),'mean_steps':float(lengths[j].float().mean()),'lengths':lengths[j].tolist(),'tail_max_abs_state':tail[j].amax(0).tolist(),'max_abs_angle':float(self.max_angle[j].max()),'max_abs_position':float(self.max_pos[j].max())})
        return rows

class EvalEngine:
    def __init__(self,a,episodes=64):
        self.e=Evaluator(a,episodes)
        f=torch.compile(self.e.step,fullgraph=True,mode='max-autotune-no-cudagraphs')
        f();torch.cuda.synchronize();self.g=capture(f)
    def run(self,horizon=10000,split=0):
        self.e.reset(split);torch.cuda.synchronize();start=time.perf_counter()
        for t in range(horizon):
            if t==max(0,horizon-1000):self.e.tail_gate.fill_(1)
            self.g.replay()
        torch.cuda.synchronize()
        return self.e.result(horizon),time.perf_counter()-start

def state_tensors(r):
    out=[]
    def visit(x):
        if torch.is_tensor(x):out.append(x)
        elif isinstance(x,(list,tuple)):
            for y in x:visit(y)
        elif isinstance(x,(Packed,PackedRunner)):
            for y in vars(x).values():visit(y)
    visit(r);return out

def save_checkpoint(r,path,meta):
    a=r.agent
    torch.save({'meta':meta,'seeds':a.seeds,'agent':{k:[t.cpu() for t in getattr(a,k)] for k in ['actor','critic','target_actor','target_critic','m','v','critic2','target_critic2'] if hasattr(a,k)},'t':a.t.cpu(),'active':a.active.cpu(),'runner':{k:getattr(r,k).cpu() for k in ['tick','state','age','replay','ptr','size','last_loss']}},path)

def load_checkpoint(r,path):
    data=torch.load(path,map_location='cpu',weights_only=False);a=r.agent
    assert data['seeds']==a.seeds
    for k,v in data['agent'].items():
        for t,x in zip(getattr(a,k),v):t.copy_(x)
    a.t.copy_(data['t']);a.active.copy_(data['active'])
    for k,v in data['runner'].items():getattr(r,k).copy_(v)
    return data['meta']

@torch.no_grad()
def main(args):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    seeds=list(range(args.seed_start,args.seed_start+args.seeds))
    wall_start=time.perf_counter();agent_type=PackedTD3 if args.algo=='td3' else Packed;a=agent_type(seeds,lr_actor=args.lr_actor,lr_critic=args.lr_critic,**({'symmetry':args.symmetry} if args.algo=='td3' else {}));r=PackedRunner(a,args.envs,args.batch,args.capacity,args.episode_limit)
    done=0;history=[];train_seconds=0.;stopped_updates={}
    if args.resume:
        meta=load_checkpoint(r,args.resume);done=meta['updates'];history=meta['history'];train_seconds=meta['train_seconds'];stopped_updates=meta.get('stopped_updates',{})
        if args.freeze_passed:
            for j,row in enumerate(history[-1]['results']):
                if row['survived']==args.eval_episodes:
                    a.active[j].zero_();stopped_updates.setdefault(str(row['seed']),done)
    else:
        for _ in range(math.ceil(args.warmup/args.envs)):r.collect(True)
    # Snapshot all mutable tensors so compilation/capture are not hidden training.
    saved=[(t,t.clone()) for t in state_tensors(r)]
    if args.execution=='staged':
        prepare=torch.compile(r.prepare,fullgraph=True,mode='max-autotune-no-cudagraphs')
        learn=torch.compile(r.learn,fullgraph=True,mode='max-autotune-no-cudagraphs')
        def f():
            index,packed=prepare()
            r.replay.scatter_(1,index,packed)
            learn()
    else:f=torch.compile(r.step,fullgraph=True,mode='max-autotune-no-cudagraphs')
    f();torch.cuda.synchronize();g=capture(f)
    for t,v in saved:t.copy_(v)
    del saved
    evaluator=EvalEngine(a,args.eval_episodes)
    torch.cuda.synchronize();setup_seconds=time.perf_counter()-wall_start
    print(json.dumps({'event':'ready','setup_seconds':setup_seconds,'hardware':hardware(),'seeds':len(seeds)}),flush=True)
    while done<args.updates:
        target=min(done+args.eval_every,args.updates);torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(target-done):g.replay()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;train_seconds+=elapsed;done=target
        rows,eval_seconds=evaluator.run(args.horizon,0)
        passed=sum(x['survived']==args.eval_episodes for x in rows)
        entry={'updates':done,'execution':args.execution,'train_seconds':train_seconds,'block_seconds':elapsed,'eval_seconds':eval_seconds,'passed':passed,'seeds':len(seeds),'worst_steps':min(x['minimum_steps'] for x in rows),'results':rows}
        history.append(entry)
        if args.freeze_passed:
            for j,row in enumerate(rows):
                if row['survived']==args.eval_episodes:
                    a.active[j].zero_();stopped_updates.setdefault(str(row['seed']),done)
        print(json.dumps({k:v for k,v in entry.items() if k!='results'}),flush=True)
        if not bool(torch.isfinite(r.last_loss).all()):raise RuntimeError('nonfinite loss')
        data={'config':vars(args),'hardware':hardware(),'setup_seconds':setup_seconds,'train_seconds':train_seconds,'wall_seconds':time.perf_counter()-wall_start,'history':history,'final':rows,'stopped_updates':stopped_updates,'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
        (out/'summary.json').write_text(json.dumps(data,indent=2))
        save_checkpoint(r,out/'checkpoint.pt',{'updates':done,'history':history,'train_seconds':train_seconds,'stopped_updates':stopped_updates})
        if args.freeze_passed and passed==len(seeds):break
    # Held-out initial-state stream is evaluated only after the fixed training run.
    heldout,heldout_seconds=evaluator.run(args.horizon,1)
    data['heldout']=heldout;data['heldout_eval_seconds']=heldout_seconds
    data['wall_seconds']=time.perf_counter()-wall_start
    data['actors']={str(seed):[w[j].squeeze(0).cpu().tolist() if i%2 else w[j].cpu().tolist() for i,w in enumerate(a.actor)] for j,seed in enumerate(seeds)}
    (out/'summary.json').write_text(json.dumps(data,indent=2))
    print(json.dumps({'event':'complete','passed':sum(x['survived']==args.eval_episodes for x in heldout),'seeds':len(seeds),'train_seconds':train_seconds,'wall_seconds':data['wall_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=100);p.add_argument('--algo',choices=['ddpg','td3'],default='ddpg');p.add_argument('--seed-start',type=int,default=1);p.add_argument('--envs',type=int,default=32);p.add_argument('--batch',type=int,default=256);p.add_argument('--capacity',type=int,default=50000);p.add_argument('--warmup',type=int,default=1024);p.add_argument('--episode-limit',type=int,default=200);p.add_argument('--updates',type=int,default=100000);p.add_argument('--eval-every',type=int,default=25000);p.add_argument('--horizon',type=int,default=10000);p.add_argument('--eval-episodes',type=int,default=64);p.add_argument('--lr-actor',type=float,default=1e-4);p.add_argument('--lr-critic',type=float,default=1e-3);p.add_argument('--resume');p.add_argument('--symmetry',action='store_true');p.add_argument('--execution',choices=['staged','full'],default='staged');p.add_argument('--freeze-passed',action='store_true');p.add_argument('--out',default='results/seeds100_ddpg');main(p.parse_args())
