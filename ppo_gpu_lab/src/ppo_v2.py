"""V2: functional flat-parameter PPO/Adam and batched critic evaluation.
Same FP32, architecture, objectives, samples, minibatches and PPO epochs as V1.
"""
import math,torch
from torch.nn import functional as F
from ppo import Trainer,physics,logprob,LOG2PI

def actor(w,s):
    h=F.relu(F.linear(s,w[:96].view(16,6),w[96:112]))
    h=F.relu(F.linear(h,w[112:368].view(16,16),w[368:384]))
    out=F.linear(h,w[384:416].view(2,16),w[416:418])
    return out[...,0],out[...,1].clamp(-5,2)

def critic(w,s):
    h=F.relu(F.linear(s,w[418:514].view(16,6),w[514:530]))
    h=F.relu(F.linear(h,w[530:786].view(16,16),w[786:802]))
    return F.linear(h,w[802:818].view(1,16),w[818:819]).squeeze(-1)

def objective(w,obs,raw,oldlp,adv,ret,idx):
    s=obs[idx];a=raw[idx];olp=oldlp[idx];ad=adv[idx];rt=ret[idx]
    mean,ls=actor(w,s);value=critic(w,s)
    lp=logprob(a,mean,ls);diff=lp-olp;ratio=diff.clamp(-50,50).exp()
    entropy=(ls+.5*(1+LOG2PI)).mean()
    aloss=-torch.minimum(ratio*ad,ratio.clamp(.8,1.2)*ad).mean()-.01*entropy
    closs=.5*(value-rt).square().mean()
    metrics=torch.stack((aloss,closs,entropy,((ratio-1)-diff).mean()))
    return aloss+closs,metrics

grad_objective=torch.func.grad_and_value(objective,has_aux=True)

@torch.no_grad()
def flat_step(w,m,v,step,lr,obs,raw,oldlp,adv,ret,idx,metrics):
    grad,(loss,measure)=grad_objective(w,obs,raw,oldlp,adv,ret,idx)
    step.add_(1)
    m.lerp_(grad,.1)
    v.mul_(.999).addcmul_(grad,grad,value=.001)
    bc1=1-.9**step;bc2=1-.999**step
    # Match PyTorch Adam: epsilon is added after bias-correcting sqrt(v).
    w.add_(-lr*(m/bc1)/((v/bc2).sqrt()+1e-8))
    metrics.copy_(measure)

class TrainerV2(Trainer):
    def __init__(self,*args,optimizer_v2=True,rollout_v2=True,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.device=='cuda' and self.mode=='graph'
        self.optimizer_v2=optimizer_v2;self.rollout_v2=rollout_v2
        self.flat=torch.cat([p.detach().flatten() for p in self.agent.parameters()])
        offset=0
        for p in self.agent.parameters():
            p.data=self.flat[offset:offset+p.numel()].view_as(p);offset+=p.numel()
        assert offset==819
        self.m=torch.zeros_like(self.flat);self.v=torch.zeros_like(self.flat);self.adam_step=torch.zeros((),device='cuda')
        self.lr=torch.cat((torch.full((418,),3e-4,device='cuda'),torch.full((401,),1e-3,device='cuda')))
        self.nextobs=torch.empty_like(self.obs)
        self.update_fn=torch.compile(flat_step,fullgraph=True)
        self.sample_fn=torch.compile(self.sample_step,fullgraph=True)
        self.values_fn=torch.compile(self.bulk_values,fullgraph=True)

    @torch.no_grad()
    def sample_step(self,noise,reset,obs,raw_out,lp_out,rew_out,term_out,end_out,nextobs):
        obs.copy_(self.state)
        mean,ls=actor(self.flat,self.state)
        raw=mean+ls.exp()*noise
        ns,r,terminal=physics(self.state,raw.clamp(-1,1),self.physical)
        self.age.add_(1)
        ended=terminal|(self.age>=1000)
        raw_out.copy_(raw);lp_out.copy_(logprob(raw,mean,ls));rew_out.copy_(r)
        term_out.copy_(terminal);end_out.copy_(ended);nextobs.copy_(ns)
        self.state.copy_(torch.where(ended[:,None],(reset-.5)*self.scale,ns))
        self.age.masked_fill_(ended,0)

    @torch.no_grad()
    def bulk_values(self):
        # Critic weights are constant throughout rollout; this is equivalent to
        # evaluating V(s_t) and V(s_{t+1}) at every environment step.
        self.val.copy_(critic(self.flat,self.obs.reshape(-1,6)).view(self.t,self.n))
        self.nextval.copy_(critic(self.flat,self.nextobs.reshape(-1,6)).view(self.t,self.n))

    @torch.no_grad()
    def collect(self):
        if not self.rollout_v2:return super().collect()
        noise=torch.randn(self.t,self.n,device=self.device)
        reset=torch.rand(self.t,self.n,6,device=self.device)
        for t in range(self.t):
            self.sample_fn(noise[t],reset[t],self.obs[t],self.raw[t],self.lp[t],self.rew[t],self.term[t],self.end[t],self.nextobs[t])
        self.values_fn()

    def iteration(self):
        if not self.optimizer_v2:return super().iteration()
        self.collect();self.gae_fn()
        obs=self.obs.reshape(-1,6);raw=self.raw.flatten();lp=self.lp.flatten();adv=self.adv.flatten();ret=self.ret.flatten()
        for _ in range(self.epochs):
            idx=torch.randperm(self.b,device=self.device)
            for start in range(0,self.b,self.mb):
                self.update_fn(self.flat,self.m,self.v,self.adam_step,self.lr,obs,raw,lp,adv,ret,idx[start:start+self.mb],self.metrics)

    def prepare(self):
        super().prepare()
        self.m.zero_();self.v.zero_();self.adam_step.zero_()
