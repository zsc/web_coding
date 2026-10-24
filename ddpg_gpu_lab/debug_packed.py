import torch,copy,json
from multiseed import *
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
a=Packed([1,2,3]);ref=copy.deepcopy(a)
torch.manual_seed(321);s=torch.randn(3,64,4,dtype=torch.float64)*.1;act=torch.randn(3,64,1,dtype=torch.float64);r=torch.randn(3,64,1,dtype=torch.float64);b=tuple(t.cuda().float() for t in (s,act,r,s+.01,torch.zeros_like(r)))
fn=torch.compile(a.update,fullgraph=True,mode='max-autotune-no-cudagraphs');saved=[(t,t.clone()) for t in state_tensors(a)];fn(b);torch.cuda.synchronize();g=capture(lambda:fn(b))
for t,v in saved:t.copy_(v)
for i in range(10):
 g.replay();ref.update(b);torch.cuda.synchronize();print(i+1,max(float((t-u).abs().max()) for t,u in zip(a.actor,ref.actor)),max(float((t-u).abs().max()) for t,u in zip(a.critic,ref.critic)),float(a.t),float(ref.t),flush=True)
