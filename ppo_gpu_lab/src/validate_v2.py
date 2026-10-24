import torch,json,pathlib
from ppo import Agent,ppo_loss,physics,logprob,Trainer
from ppo_v2 import TrainerV2,flat_step,actor,critic
root=pathlib.Path(__file__).resolve().parents[1];torch.set_num_threads(1);torch.manual_seed(456)
a=Agent().cuda();w=torch.cat([p.detach().flatten() for p in a.parameters()]);m=torch.zeros_like(w);v=torch.zeros_like(w);step=torch.zeros((),device='cuda');lr=torch.cat((torch.full((418,),3e-4,device='cuda'),torch.full((401,),1e-3,device='cuda')))
opt=torch.optim.Adam([{'params':a.actor.parameters(),'lr':3e-4},{'params':a.critic.parameters(),'lr':1e-3}],capturable=True,fused=True)
s=torch.randn(2048,6,device='cuda')*.3
with torch.no_grad():
 mean,ls,value=a(s);raw=mean+ls.exp()*torch.randn_like(mean);lp=logprob(raw,mean,ls);adv=torch.randn_like(raw);ret=torch.randn_like(raw)*20
out={};metrics=torch.empty(4,device='cuda');compiled=torch.compile(flat_step,fullgraph=True)
# Same fixed samples/permutation, 20 consecutive Adam updates including moments.
errors=[]
for i in range(20):
 idx=torch.randperm(2048,device='cuda')[:512]
 opt.zero_grad(set_to_none=True);loss,met=ppo_loss(a,s[idx],raw[idx],lp[idx],adv[idx],ret[idx]);loss.backward();opt.step()
 compiled(w,m,v,step,lr,s,raw,lp,adv,ret,idx,metrics)
 errors.append((w-torch.cat([p.detach().flatten() for p in a.parameters()])).abs().max().item())
out['adam_20_updates_max_abs']=max(errors);out['metrics_max_abs']=(met-metrics).abs().max().item()
assert out['adam_20_updates_max_abs']<2e-5,out
# Forced episode boundaries and identical supplied noise, across both dynamics.
for physical in (False,True):
 t=TrainerV2(n=32,horizon=16,minibatch=512,epochs=1,physical=physical)
 state=t.state.clone();age=torch.arange(980,1012,device='cuda');t.age.copy_(age)
 noise=torch.randn(16,32,device='cuda');reset=torch.rand(16,32,6,device='cuda')
 expected={k:[] for k in ['obs','raw','lp','rew','term','end','nextobs','val','nextval']}
 with torch.no_grad():
  for j in range(16):
   mean,ls,value=t.agent(state);raw=mean+ls.exp()*noise[j];ns,r,term=physics(state,raw.clamp(-1,1),physical);age+=1;end=term|(age>=1000)
   for k,val in dict(obs=state,raw=raw,lp=logprob(raw,mean,ls),rew=r,term=term,end=end,nextobs=ns,val=value,nextval=t.agent.critic(ns).squeeze(-1)).items():expected[k].append(val.clone())
   state=torch.where(end[:,None],(reset[j]-.5)*t.scale,ns);age.masked_fill_(end,0)
  for j in range(16):t.sample_fn(noise[j],reset[j],t.obs[j],t.raw[j],t.lp[j],t.rew[j],t.term[j],t.end[j],t.nextobs[j])
  t.values_fn()
 errors={k:(getattr(t,k).float()-torch.stack(vals).float()).abs().max().item() for k,vals in expected.items()}
 errors['state']=(t.state-state).abs().max().item();errors['age']=(t.age-age).abs().max().item()
 out['rollout_physical' if physical else 'rollout_source']=errors
 assert errors['term']==0 and errors['end']==0 and errors['age']==0,out
 assert max(errors.values())<2e-4,out
out['passed']=True;(root/'results/validation_v2.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
