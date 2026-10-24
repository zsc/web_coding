import argparse,copy,json,math
from pathlib import Path
import torch
from ddpg import Agent,Runner,forward,backward,physics,compiled_update,capture,tensors,hardware
p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');p.add_argument('--out',default='results/validation.json');args=p.parse_args()
torch.set_num_threads(1)
f=json.loads(Path('results/fixture.json').read_text())
def make(device='cpu',dtype=torch.float64):
    a=Agent(device,dtype,optimizer='sgd')
    for k,values in f['before'].items():
        for t,v in zip(getattr(a,k),values):t.copy_(torch.tensor(v,device=device,dtype=dtype))
    b=tuple(torch.tensor([e[k] if k in ['s','a','ns'] else [float(e[k])] for e in f['batch']],device=device,dtype=dtype) for k in ['s','a','r','ns','d'])
    return a,b
checks=[]
def check(name,actual,expected,atol=1e-10,rtol=1e-9):
    expected=torch.as_tensor(expected,device=actual.device,dtype=actual.dtype)
    torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
    checks.append({'name':name,'max_abs_error':float((actual-expected).abs().max()),'atol':atol,'rtol':rtol,'passed':True})
a,b=make();s,act,r,ns,d=b
check('actor vs original JS',forward(a.actor,s,True)[0],[e['a'] for e in f['predictions']])
q,c=forward(a.critic,torch.cat((s,act),1));_,dx=backward(a.critic,c,torch.ones_like(q))
check('critic vs original JS',q,[[e['q']] for e in f['predictions']]);check('dQ/da vs original JS',dx[:,-1:],[e['dqda'] for e in f['predictions']])
ns2,r2,d2=physics(s,act)
check('physics vs original JS',ns2,[e['nextState'] for e in f['physics']]);check('reward vs original JS',r2,[[e['reward']] for e in f['physics']]);check('terminal vs original JS',d2.double(),[[float(e['done'])] for e in f['physics']])
loss=a.serial_update(b)
check('serial losses vs original JS',loss,[f['losses']['criticLoss'],f['losses']['actorLoss']])
for k,v in f['after'].items():check('serial '+k+' weights vs original JS',torch.cat([t.flatten() for t in getattr(a,k)]),torch.cat([torch.tensor(t,dtype=torch.float64).flatten() for t in v]))
# Independent autograd reference for both manual neural-network backward paths.
for actor in [False,True]:
    a,b=make();w=a.actor if actor else a.critic;x=b[0] if actor else torch.cat((b[0],b[1]),1)
    wr=[t.clone().requires_grad_() for t in w];xr=x.clone().requires_grad_();y,c=forward(w,x,actor);dy=torch.randn_like(y);g,gx=backward(w,c,dy,actor)
    yr,_=forward(wr,xr,actor);ref=torch.autograd.grad((yr*dy).sum(),wr+[xr]);check(('actor' if actor else 'critic')+' analytic gradients vs autograd',torch.cat([t.flatten() for t in g+[gx]]),torch.cat([t.flatten() for t in ref]))
a,b=make();a2=copy.deepcopy(a);a.update(tuple(x[:1] for x in b));a2.serial_update(tuple(x[:1] for x in b))
check('batch=1 preserves serial SGD',torch.cat([t.flatten() for k in ['actor','critic'] for t in getattr(a,k)]),torch.cat([t.flatten() for k in ['actor','critic'] for t in getattr(a2,k)]))
# Boundary semantics: time limit resets but must not zero bootstrap mask.
a=Agent();runner=Runner(a,n=8,capacity=10,batch_size=4);runner.age.fill_(199);runner.state.zero_();runner.collect()
assert not bool(runner.replay[:8,-1].any()) and int(runner.age.max())==0
for _ in range(3):runner.collect()
assert runner.ptr.item()==2 and runner.size.item()==10
checks.append({'name':'replay wraparound + nonterminal time-limit reset','passed':True})
# Check the hand-written Adam update against the library optimizer.
a=Agent('cpu',torch.float64,optimizer='adam');ref=[t.clone().requires_grad_() for t in a.critic]
opt=torch.optim.Adam(ref,lr=1e-3,betas=(0.9,0.999),eps=1e-8)
for _ in range(5):
    grads=[torch.randn_like(t) for t in a.critic]
    a.t.add_(1);a.apply(a.critic,grads,1e-3,0)
    for t,g in zip(ref,grads):t.grad=g.clone()
    opt.step()
check('manual Adam vs torch.optim.Adam, 5 updates',torch.cat([t.flatten() for t in a.critic]),torch.cat([t.detach().flatten() for t in ref]))
if args.cuda:
    torch.backends.cuda.matmul.allow_tf32=False
    a,b=make('cuda',torch.float32);a2=copy.deepcopy(a);step=compiled_update(a,True)
    saved=[(t,t.clone()) for t in tensors(a)];step(b);torch.cuda.synchronize();graph=capture(lambda:step(b))
    for t,v in saved:t.copy_(v)
    for _ in range(10):graph.replay();a2.update(b)
    torch.cuda.synchronize()
    check('compiled CUDA Graph vs eager, 10 updates',torch.cat([t.flatten() for k in ['actor','critic','target_actor','target_critic'] for t in getattr(a,k)]),torch.cat([t.flatten() for k in ['actor','critic','target_actor','target_critic'] for t in getattr(a2,k)]),atol=3e-5,rtol=3e-4)
    a=Agent('cuda',optimizer='adam');a2=copy.deepcopy(a)
    step=compiled_update(a,True);saved=[(t,t.clone()) for t in tensors(a)]
    step(b);torch.cuda.synchronize();graph=capture(lambda:step(b))
    for t,v in saved:t.copy_(v)
    for _ in range(10):graph.replay();a2.update(b)
    torch.cuda.synchronize()
    check('compiled Adam CUDA Graph vs eager, 10 updates',torch.cat([t.flatten() for k in ['actor','critic'] for t in getattr(a,k)]),torch.cat([t.flatten() for k in ['actor','critic'] for t in getattr(a2,k)]),atol=3e-5,rtol=3e-4)
    # Full graph must advance RNG, replay pointer and sample size each replay.
    ar=Agent('cuda');rr=Runner(ar,n=8,capacity=1000,batch_size=64)
    for _ in range(10):rr.collect(True)
    graph=capture(rr.step);ptr=rr.ptr.item();state=rr.state.clone()
    for _ in range(4):graph.replay()
    torch.cuda.synchronize();assert rr.ptr.item()==(ptr+32)%1000;assert not torch.equal(state,rr.state)
    checks.append({'name':'full graph advances replay, physics and random state','passed':True})
Path(args.out).write_text(json.dumps({'hardware':hardware(),'checks':checks},indent=2));print(json.dumps(checks,indent=2))
