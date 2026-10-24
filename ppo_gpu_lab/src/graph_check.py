"""Compare a replayed compiled CUDA minibatch update with eager autograd/Adam."""
import torch,json,pathlib,copy
from ppo import Agent,ppo_loss
torch.manual_seed(11);torch.set_num_threads(1)
a=Agent().cuda();b=Agent().cuda();b.load_state_dict(a.state_dict())
s=torch.randn(64,6,device='cuda')*.1
with torch.no_grad():
    m,l,v=a(s);raw=m+l.exp()*torch.randn_like(m);lp=-.5*((raw-m)*(-l).exp()).square()-l-.5*1.8378770664093453
    adv=torch.randn_like(m);ret=torch.randn_like(m)
def opt(model):return torch.optim.Adam([{'params':model.actor.parameters(),'lr':3e-4},{'params':model.critic.parameters(),'lr':1e-3}],capturable=True,foreach=False,fused=True)
oa,ob=opt(a),opt(b)
fn=torch.compile(lambda:ppo_loss(a,s,raw,lp,adv,ret),fullgraph=True)
def step(model,opt,compiled=False):
    opt.zero_grad(set_to_none=True)
    loss,metrics=fn() if compiled else ppo_loss(model,s,raw,lp,adv,ret)
    loss.backward();opt.step()
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(2):step(a,oa,True);step(b,ob)
torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
# Reset both to the same weights and moments before capture.
b.load_state_dict(a.state_dict());ob.load_state_dict(copy.deepcopy(oa.state_dict()))
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):step(a,oa,True)
g.replay();step(b,ob);torch.cuda.synchronize()
err=max((x-y).abs().max().item() for x,y in zip(a.parameters(),b.parameters()))
result={'compiled_graph_vs_eager_one_adam_update_max_abs':err,'passed':err<2e-6}
assert result['passed'],result
p=pathlib.Path(__file__).resolve().parents[1]/'results/graph_validation.json';p.write_text(json.dumps(result,indent=2));print(result)
