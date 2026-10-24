import copy,json
from pathlib import Path
import torch
from multiseed import PackedTD3,fw,bw,uniform,state_tensors,capture
# Independent autograd reference for target construction, delayed updates,
# observation/action scaling and per-seed gradient clipping.
torch.set_num_threads(1);checks=[]
a=PackedTD3([1001,1002],device='cpu',dtype=torch.float64)
refs=[]
for j in range(2):
 nets={k:[t[j].clone().requires_grad_(k in ['actor','critic','critic2']) for t in getattr(a,k)] for k in ['actor','critic','critic2','target_actor','target_critic','target_critic2']}
 opts={k:torch.optim.Adam(nets[k],lr=1e-4 if k=='actor' else 1e-3) for k in ['actor','critic','critic2']};refs.append((nets,opts))
torch.manual_seed(789);s=torch.randn(2,32,4,dtype=torch.float64)*.1;act=torch.randn(2,32,1,dtype=torch.float64);r=torch.rand(2,32,1,dtype=torch.float64);b=(s,act,r,s+.01,torch.zeros_like(r))
scale=a.obs_scale[0]
def ff(w,x,actor=False):
 x=(x@w[0].T+w[1]).relu();x=(x@w[2].T+w[3]).relu();x=x@w[4].T+w[5];return x.tanh()*10 if actor else x
for step in range(4):
 idx=torch.arange(32)[None,:];u1=uniform(a.seedkey,a.t.long(),idx,11).clamp(min=1e-7);u2=uniform(a.seedkey,a.t.long(),idx,12);noise=((-2*u1.log()).sqrt()*(2*torch.pi*u2).cos()*2).clamp(-5,5).unsqueeze(-1)
 a.update(b)
 for j,(n,o) in enumerate(refs):
  st,ac,rw,ns,dn=(t[j] for t in b)
  with torch.no_grad():
   na=(ff(n['target_actor'],ns*scale,True)+noise[j]).clamp(-10,10);nx=torch.cat((ns*scale,na/10),1);y=rw+.99*(1-dn)*torch.minimum(ff(n['target_critic'],nx),ff(n['target_critic2'],nx))
  for k in ['critic','critic2']:
   o[k].zero_grad();q=ff(n[k],torch.cat((st*scale,ac/10),1));loss=.5*(q-y).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(n[k],10);o[k].step()
  if (step+1)%2==0:
   o['actor'].zero_grad();aa=ff(n['actor'],st*scale,True);loss=-ff(n['critic'],torch.cat((st*scale,aa/10),1)).mean();loss.backward();torch.nn.utils.clip_grad_norm_(n['actor'],10);o['actor'].step()
   with torch.no_grad():
    for k in ['actor','critic','critic2']:
     for target,online in zip(n['target_'+k],n[k]):target.lerp_(online,.005)
 for j,(n,o) in enumerate(refs):
  for k,w in n.items():
   x=torch.cat([t[j].flatten() for t in getattr(a,k)]);y=torch.cat([t.detach().flatten() for t in w]);torch.testing.assert_close(x,y,atol=2e-10,rtol=1e-8)
checks.append({'name':'TD3 4 updates vs independent autograd Adam reference, 2 seeds','passed':True})
# Frozen learner must keep every policy/value/target weight unchanged.
a.active[0].zero_();saved={k:[w[0].clone() for w in getattr(a,k)] for k in ['actor','critic','critic2','target_actor','target_critic','target_critic2']}
for _ in range(4):a.update(b)
for k,values in saved.items():
 for w,v in zip(getattr(a,k),values):assert torch.equal(w[0],v)
checks.append({'name':'all frozen learner weights remain unchanged','passed':True})
Path('results/validation_td3.json').write_text(json.dumps(checks,indent=2));print(checks)
if '--cuda' in __import__('sys').argv:
 a=PackedTD3([1001,1002]);e=copy.deepcopy(a);bb=tuple(t.float().cuda() for t in b)
 tensors=state_tensors(a);snap=[t.clone() for t in tensors]
 f=torch.compile(a.update,fullgraph=True,mode='max-autotune-no-cudagraphs');f(bb);torch.cuda.synchronize();g=capture(lambda:f(bb))
 for t,v in zip(tensors,snap):t.copy_(v)
 for _ in range(4):g.replay();e.update(bb)
 torch.cuda.synchronize();err=0.
 for key in ['actor','critic','critic2','target_actor','target_critic','target_critic2']:
  for x,y in zip(getattr(a,key),getattr(e,key)):
   err=max(err,float((x-y).abs().max()));torch.testing.assert_close(x,y,atol=3e-6,rtol=3e-5)
 checks.append({'name':'CUDA compiled graph vs eager TD3, 4 updates including actor delay','passed':True,'max_abs_error':err})
 Path('results/validation_td3.json').write_text(json.dumps(checks,indent=2));print(checks[-1])
