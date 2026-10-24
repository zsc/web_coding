"""Device-resident PPO for the source double pendulum; optional physical equations.
Source equations are deliberately preserved for performance/correctness comparison.
"""
import math
import torch
from torch import nn

LOG2PI = math.log(2 * math.pi)
RESET_SCALE = [.1, .1, .02, .05, .02, .05]


def physics(state, action, physical=False):
    x, v, t1, w1, t2, w2 = state.unbind(-1)
    c1, c2, c12 = t1.cos(), t2.cos(), (t1-t2).cos()
    s1, s2, s12 = t1.sin(), t2.sin(), (t1-t2).sin()
    a = 1.2
    b = .15*c1 if physical else .15*c1+.05*c2
    c = .05*c2
    d = torch.full_like(x,.1275) if physical else .1525+.1*c12
    e = .05*c12 if physical else .05*c12+.025
    f = .0275
    n0 = action*10 + .15*s1*w1.square()+.05*s2*w2.square()
    if physical:
        n1 = 1.4715*s1-.05*s12*w2.square()
        n2 = .4905*s2+.05*s12*w1.square()
    else:
        n0 = n0+.05*s12*w2.square()
        n1 = -1.4715*s1-.05*s12*w2*(w1+w2)
        n2 = -.4905*s2+.05*s12*w1.square()
    # Batched explicit LDL^T solve of the symmetric 3x3 mass matrix.
    l10,l20=b/a,c/a
    d1=d-b*l10
    l21=(e-c*l10)/d1
    d2=f-c*l20-l21.square()*d1
    y0=n0
    y1=n1-l10*y0
    y2=n2-l20*y0-l21*y1
    z2=y2/d2
    z1=y1/d1-l21*z2
    z0=y0/a-l10*z1-l20*z2
    nv,nw1,nw2=v+.02*z0,w1+.02*z1,w2+.02*z2
    nx,nt1,nt2=x+.02*nv,t1+.02*nw1,t2+.02*nw2
    # fmod matches JavaScript remainder even outside the principal interval.
    nt1=torch.fmod(nt1+math.pi,2*math.pi)-math.pi
    nt2=torch.fmod(nt2+math.pi,2*math.pi)-math.pi
    ns=torch.stack((nx,nv,nt1,nw1,nt2,nw2),-1)
    done=(nx.abs()>2.4)|(nt1.abs()>math.pi/15)|(nt2.abs()>math.pi/15)
    finite=torch.isfinite(ns).all(-1)
    reward=1-.5*(nt1.square()+nt2.square())-.1*nx.square()-.01*(nv.square()+nw1.square()+nw2.square())
    reward=torch.where(done,-100.,reward)
    reward=torch.where(finite,reward,-200.)
    return ns,reward,done|(~finite)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU(),nn.Linear(16,2))
        self.critic=nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU(),nn.Linear(16,1))
        for net in [self.actor,self.critic]:
            for i,layer in enumerate([net[0],net[2],net[4]]):
                scale=math.sqrt((1 if i==2 else 2)/layer.in_features)
                nn.init.uniform_(layer.weight,-scale,scale)
                nn.init.zeros_(layer.bias)
        with torch.no_grad(): self.actor[-1].bias[1]=math.log(.5)

    def forward(self,s):
        out=self.actor(s)
        return out[:,0],out[:,1].clamp(-5,2),self.critic(s).squeeze(-1)


def logprob(raw,mean,ls):
    return -.5*((raw-mean)*(-ls).exp()).square()-ls-.5*LOG2PI


def ppo_loss(agent,s,raw,oldlp,adv,ret,legacy=False):
    mean,ls,value=agent(s)
    lp=logprob(raw,mean,ls)
    ratio=(lp-oldlp).clamp(-50,50).exp()
    policy=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
    entropy=(ls+.5*(1+LOG2PI)).mean()
    actor=policy+(.01*entropy if legacy else -.01*entropy)
    critic=.5*(value-ret).square().mean()
    return actor+critic,torch.stack((actor.detach(),critic.detach(),entropy.detach(),((ratio-1)-(lp-oldlp)).mean().detach()))


class Trainer:
    def __init__(self,device='cuda',mode='graph',n=256,horizon=64,minibatch=2048,epochs=10,seed=7,physical=False):
        torch.manual_seed(seed)
        self.device=device; self.n=n; self.t=horizon; self.b=n*horizon; self.mb=minibatch;self.epochs=epochs;self.physical=physical
        assert self.b%minibatch==0
        self.agent=Agent().to(device)
        self.optimizer=torch.optim.Adam([{'params':self.agent.actor.parameters(),'lr':3e-4},{'params':self.agent.critic.parameters(),'lr':1e-3}],capturable=device=='cuda',foreach=False,fused=device=='cuda')
        self.scale=torch.tensor(RESET_SCALE,device=device)
        self.state=(torch.rand(n,6,device=device)-.5)*self.scale
        self.age=torch.zeros(n,device=device,dtype=torch.long)
        self.obs=torch.empty(horizon,n,6,device=device)
        self.raw=torch.empty(horizon,n,device=device)
        self.lp=torch.empty_like(self.raw);self.val=torch.empty_like(self.raw);self.rew=torch.empty_like(self.raw)
        self.term=torch.empty_like(self.raw);self.end=torch.empty_like(self.raw);self.nextval=torch.empty_like(self.raw)
        self.adv=torch.empty_like(self.raw);self.ret=torch.empty_like(self.raw)
        self.metrics=torch.zeros(4,device=device)
        self.env_fn=lambda s,a:physics(s,a,physical)
        self.loss_fn=lambda *args:ppo_loss(self.agent,*args)
        self.gae_fn=self._gae
        self.mode=mode
        if mode=='graph':
            self.env_fn=torch.compile(self.env_fn,fullgraph=True)
            self.loss_fn=torch.compile(self.loss_fn,fullgraph=True)
            self.gae_fn=torch.compile(self._gae,fullgraph=True)
        self.graph=None

    @torch.no_grad()
    def collect(self):
        for t in range(self.t):
            self.obs[t].copy_(self.state)
            mean,ls,v=self.agent(self.state)
            raw=mean+ls.exp()*torch.randn_like(mean)
            # Store latent Gaussian action; only the environment gets the clipped action.
            ns,r,terminal=self.env_fn(self.state,raw.clamp(-1,1))
            self.age.add_(1)
            ended=terminal|(self.age>=1000)
            nv=self.agent.critic(ns).squeeze(-1)
            self.raw[t].copy_(raw);self.lp[t].copy_(logprob(raw,mean,ls));self.val[t].copy_(v);self.rew[t].copy_(r)
            self.term[t].copy_(terminal);self.end[t].copy_(ended);self.nextval[t].copy_(nv)
            self.state.copy_(torch.where(ended[:,None],(torch.rand_like(ns)-.5)*self.scale,ns))
            self.age.masked_fill_(ended,0)

    def _gae(self):
        last=torch.zeros_like(self.rew[0])
        for t in range(self.t-1,-1,-1):
            delta=self.rew[t]+.99*self.nextval[t]*(1-self.term[t])-self.val[t]
            last=delta+.99*.95*(1-self.end[t])*last
            self.adv[t].copy_(last)
        self.ret.copy_(self.adv+self.val)
        self.adv.copy_((self.adv-self.adv.mean())/(self.adv.std(correction=0)+1e-8))

    def iteration(self):
        self.collect()
        self.gae_fn()
        obs=self.obs.reshape(-1,6); raw=self.raw.flatten();lp=self.lp.flatten();adv=self.adv.flatten();ret=self.ret.flatten()
        for _ in range(self.epochs):
            idx=torch.randperm(self.b,device=self.device)
            for start in range(0,self.b,self.mb):
                j=idx[start:start+self.mb]
                self.optimizer.zero_grad(set_to_none=True)
                loss,metrics=self.loss_fn(obs[j],raw[j],lp[j],adv[j],ret[j])
                loss.backward()
                self.optimizer.step()
                self.metrics.copy_(metrics)

    def _prepare(self):
        if self.mode!='graph':
            self.iteration()
            return
        # Side-stream warmup, capture entire rollout + GAE + shuffled minibatches + Adam.
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):self.iteration()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.iteration()
        torch.cuda.synchronize()

    def prepare(self):
        # Warmup may update parameters. Restore all training state in-place so
        # benchmark/training samples start from the initial model after capture.
        weights={k:v.clone() for k,v in self.agent.state_dict().items()}
        state=self.state.clone();age=self.age.clone()
        rng=torch.cuda.get_rng_state() if self.device=='cuda' else torch.get_rng_state()
        self._prepare()
        self.agent.load_state_dict(weights)
        self.state.copy_(state);self.age.copy_(age)
        for st in self.optimizer.state.values():
            for value in st.values():
                if torch.is_tensor(value):value.zero_()
        if self.device=='cuda':torch.cuda.set_rng_state(rng)
        else:torch.set_rng_state(rng)

    def step(self):
        if self.graph is None:self.iteration()
        else:self.graph.replay()

    @torch.no_grad()
    def evaluate(self,seed=123,count=128,kind='policy',trajectory=False):
        gen=torch.Generator(device=self.device).manual_seed(seed)
        s=(torch.rand(count,6,device=self.device,generator=gen)-.5)*self.scale
        alive=torch.ones(count,device=self.device,dtype=torch.bool)
        returns=torch.zeros(count,device=self.device);lengths=torch.zeros_like(returns);frames=[]
        fn=self.env_fn
        for t in range(1000):
            a=self.agent.actor(s)[:,0].clamp(-1,1) if kind=='policy' else torch.zeros(count,device=self.device)
            ns,r,done=fn(s,a)
            returns.add_(r*alive);lengths.add_(alive)
            if trajectory:frames.append(torch.cat((s[:1],a[:1,None]),1).cpu().tolist()[0])
            alive=alive&(~done)
            s=torch.where(alive[:,None],ns,s)
        return {'kind':kind,'seed':seed,'count':count,'mean_return':returns.mean().item(),'mean_length':lengths.mean().item(),'survival_1000':alive.float().mean().item(),'returns':returns.cpu().tolist(),'lengths':lengths.cpu().tolist(),'trajectory':frames}
