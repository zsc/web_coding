import copy,json
from pathlib import Path
import torch
from ddpg import physics
from multiseed import PackedTD3,fw,bw,state_tensors,capture
checks=[];torch.set_num_threads(1)
a=PackedTD3([1001,1002],device='cpu',dtype=torch.float64,symmetry=True)
torch.manual_seed(173);s=torch.randn(2,64,4,dtype=torch.float64)*.1;act=torch.randn(2,64,1,dtype=torch.float64)
y,c=a.policy(a.actor,s);dy=torch.randn_like(y)
g=a.policy_backward(c,dy)
w=[x.clone().requires_grad_() for x in a.actor]
yref=10*(((fw(w,s*a.obs_scale,False)[0]-fw(w,-s*a.obs_scale,False)[0])*.5).tanh())
(yref*dy).sum().backward()
for x,ref in zip(g,w):torch.testing.assert_close(x,ref.grad,atol=1e-12,rtol=1e-10)
torch.testing.assert_close(a.action(-s),-a.action(s),atol=0,rtol=0)
n,r,d=physics(s.reshape(-1,4),act.reshape(-1,1));nn,rr,dd=physics(-s.reshape(-1,4),-act.reshape(-1,1))
assert torch.equal(n,-nn) and torch.equal(r,rr) and torch.equal(d,dd)
checks.append({'name':'odd actor gradient vs autograd; exact physics/reward mirror symmetry','passed':True})
a.actor[-1].sub_(20)
torch.testing.assert_close(a.action(s),y,atol=1e-12,rtol=1e-10)
assert float(a.action(s).abs().max())>1e-3
checks.append({'name':'large common output bias cannot saturate the antisymmetric actor','passed':True})
torch.backends.cuda.matmul.allow_tf32=False
a=PackedTD3([1001,1002],symmetry=True);ref=copy.deepcopy(a);b=(s.float().cuda(),act.float().cuda(),torch.rand(2,64,1,device='cuda'),s.float().cuda()+.01,torch.zeros(2,64,1,device='cuda'))
tensors=state_tensors(a);snapshot=[t.clone() for t in tensors]
f=torch.compile(a.update,fullgraph=True,mode='max-autotune-no-cudagraphs');f(b);torch.cuda.synchronize();g=capture(lambda:f(b))
for t,v in zip(tensors,snapshot):t.copy_(v)
for _ in range(4):g.replay();ref.update(b)
torch.cuda.synchronize();error=0.
for key in ['actor','critic','critic2','target_actor','target_critic','target_critic2']:
 for x,y in zip(getattr(a,key),getattr(ref,key)):
  error=max(error,float((x-y).abs().max()));print(key,float((x-y).abs().max()),flush=True);torch.testing.assert_close(x,y,atol=3e-6,rtol=3e-5)
action_error=float((a.action(b[0])-ref.action(b[0])).abs().max());torch.testing.assert_close(a.action(b[0]),ref.action(b[0]),atol=3e-4,rtol=3e-5)
checks.append({'name':'symmetric TD3 4 CUDA graph updates vs eager incl mirrored minibatch','passed':True,'max_abs_error':error,'atol':3e-6,'rtol':3e-5,'max_action_error':action_error})
Path('results/validation_symmetry.json').write_text(json.dumps(checks,indent=2));print(checks)
