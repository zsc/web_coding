import copy,json,argparse
from pathlib import Path
import torch
from ddpg import Agent,forward
from multiseed import Packed,PackedRunner,fw,uniform,state_tensors,capture
p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');args=p.parse_args();torch.set_num_threads(1)
checks=[]
def check(name,x,y,atol=1e-10,rtol=1e-9):
    torch.testing.assert_close(x,y,atol=atol,rtol=rtol);checks.append({'name':name,'max_abs_error':float((x-y).abs().max()),'passed':True,'atol':atol,'rtol':rtol})
a=Packed([1,2,3],device='cpu',dtype=torch.float64);refs=[]
for seed in a.seeds:torch.manual_seed(seed);refs.append(Agent('cpu',torch.float64))
torch.manual_seed(321);s=torch.randn(3,64,4,dtype=torch.float64)*.1;act=torch.randn(3,64,1,dtype=torch.float64);r=torch.randn(3,64,1,dtype=torch.float64);ns=s+.01;d=torch.zeros_like(r);b=s,act,r,ns,d
for _ in range(20):
    a.update(b)
    for j,ref in enumerate(refs):ref.update(tuple(t[j] for t in b))
for j,ref in enumerate(refs):
    for key in ['actor','critic','target_actor','target_critic']:
        check('seed %d %s vs separate learner'%(a.seeds[j],key),torch.cat([t[j].flatten() for t in getattr(a,key)]),torch.cat([t.flatten() for t in getattr(ref,key)]))
# Seed-dependent randomness is invariant to order and pack size.
keys=torch.tensor([[7],[16],[28]]);idx=torch.arange(257)[None,:];tick=torch.tensor(184)
u=uniform(keys,tick,idx,4);u2=uniform(keys.flip(0),tick,idx,4)
check('random stream pack/order invariance',u,u2.flip(0));assert not torch.equal(u[0],u[1]);assert bool(((u>=0)&(u<1)).all())
# Each seed has an isolated replay pool: use unique reward markers.
a=Packed([11,12,13],device='cpu');rr=PackedRunner(a,n=8,batch=16,capacity=10)
rr.size.fill_(10)
for j in range(3):rr.replay[j,:,5]=j+100
samples=rr.sample();check('replay does not mix seeds',samples[2][:,:,0],torch.arange(100,103,dtype=torch.float32)[:,None].expand(-1,16))
rr.state.zero_();rr.age.fill_(199);rr.collect();assert not rr.replay[:,:8,-1].any();assert rr.age.max()==0
checks.append({'name':'time limit remains nonterminal','passed':True})
if args.cuda:
    torch.backends.cuda.matmul.allow_tf32=False
    a=Packed([1,2,3]);ref=copy.deepcopy(a);bc=tuple(t.cuda().float() for t in b)
    fn=torch.compile(a.update,fullgraph=True,mode='max-autotune-no-cudagraphs');saved=[(t,t.clone()) for t in state_tensors(a)]
    fn(bc);torch.cuda.synchronize();g=capture(lambda:fn(bc))
    for t,v in saved:t.copy_(v)
    for i in range(10):
        g.replay();ref.update(bc)
        if i==0:
            torch.cuda.synchronize()
            check('packed compiled graph single update',torch.cat([t.flatten() for t in a.actor+a.critic]),torch.cat([t.flatten() for t in ref.actor+ref.critic]),2e-6,2e-5)
    torch.cuda.synchronize()
    check('packed compiled graph vs eager 10 updates',torch.cat([t.flatten() for t in a.actor+a.critic]),torch.cat([t.flatten() for t in ref.actor+ref.critic]),2e-4,1e-3)
    check('10-update actor action agreement',fw(a.actor,bc[0],True)[0],fw(ref.actor,bc[0],True)[0],2e-3,1e-3)
    # Full loop uses deterministic keyed RNG, allowing exact-path comparisons.
    rr=PackedRunner(Packed([1,2,3]),n=8,batch=16,capacity=128);ref=copy.deepcopy(rr)
    for _ in range(5):rr.collect(True);ref.collect(True)
    fn=torch.compile(rr.step,fullgraph=True,mode='max-autotune-no-cudagraphs');saved=[(t,t.clone()) for t in state_tensors(rr)];fn();torch.cuda.synchronize();g=capture(fn)
    for t,v in saved:t.copy_(v)
    for _ in range(5):g.replay();ref.step()
    torch.cuda.synchronize()
    check('full packed graph state vs eager',rr.state,ref.state,3e-5,3e-4)
    check('full packed graph replay vs eager',rr.replay,ref.replay,3e-5,3e-4)
    check('full packed graph actor vs eager',torch.cat([t.flatten() for t in rr.agent.actor]),torch.cat([t.flatten() for t in ref.agent.actor]),3e-5,3e-4)
Path('results/validation_multiseed.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
