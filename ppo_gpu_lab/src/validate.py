import json,torch,pathlib
from ppo import physics,Agent,ppo_loss,Trainer
root=pathlib.Path(__file__).resolve().parents[1]
f=json.loads((root/'results/fixtures.json').read_text())

def load(net,data):
    with torch.no_grad():
        for layer,obj in zip([net[0],net[2],net[4]],data):
            layer.weight.copy_(torch.tensor(obj['weights']));layer.bias.copy_(torch.tensor(obj['biases']).flatten())

out={}
for dtype in (torch.float64,torch.float32):
    s=torch.tensor([x['state'] for x in f['physics']],device='cuda',dtype=dtype)
    a=torch.tensor([x['action'] for x in f['physics']],device='cuda',dtype=dtype)
    expected=torch.tensor([x['observation'] for x in f['physics']],device='cuda',dtype=dtype)
    ns,r,d=physics(s,a)
    err=(ns-expected).abs().max().item()
    out[str(dtype)]={'state_max_abs':err,'reward_max_abs':(r-torch.tensor([x['reward'] for x in f['physics']],device='cuda',dtype=dtype)).abs().max().item(),'done_equal':d.cpu().tolist()==[x['done'] for x in f['physics']]}
    assert err<(1e-5 if dtype==torch.float32 else 1e-11)
    assert out[str(dtype)]['done_equal']
a=Agent().double();load(a.actor,f['actor']);load(a.critic,f['critic'])
s=torch.tensor([x['state'] for x in f['forward']],dtype=torch.float64)
out['forward_actor_max_abs']=(a.actor(s)-torch.tensor([x['actor'] for x in f['forward']],dtype=torch.float64)).abs().max().item()
# Weight import used default float32; this forward tolerance accounts for fixture import.
assert out['forward_actor_max_abs']<1e-7
u=f['update'];load(a.actor,u['actor']);load(a.critic,u['critic']);data=u['data']
s=torch.tensor(data['states'],dtype=torch.float64);raw=torch.tensor(data['actions'],dtype=torch.float64);lp=torch.tensor(data['old_log_probs'],dtype=torch.float64)
v=torch.tensor(data['old_values'],dtype=torch.float64);r=torch.tensor(data['rewards'],dtype=torch.float64);done=torch.tensor(data['dones'],dtype=torch.float64)
adv=torch.empty_like(v);last=0.;nv=a.critic(s[-1:]).item()
for t in range(63,-1,-1):
    delta=r[t]+.99*(nv if t==63 else v[t+1])*(1-done[t])-v[t]
    last=delta+.99*.95*(1-done[t])*last;adv[t]=last
ret=adv+v;adv=(adv-adv.mean())/(adv.std(correction=0)+1e-8)
opt=torch.optim.Adam([{'params':a.actor.parameters(),'lr':3e-4},{'params':a.critic.parameters(),'lr':1e-3}])
loss,metrics=ppo_loss(a,s,raw,lp,adv,ret,legacy=True);loss.backward();opt.step()
err=0
for net,key in [(a.actor,'actor_after'),(a.critic,'critic_after')]:
    for layer,obj in zip([net[0],net[2],net[4]],u[key]):
        err=max(err,(layer.weight-torch.tensor(obj['weights'],dtype=torch.float64)).abs().max().item(),(layer.bias-torch.tensor(obj['biases'],dtype=torch.float64).flatten()).abs().max().item())
out['legacy_one_update_weight_max_abs']=err;assert err<2e-6
# GAE: real terminal => no bootstrap; time limit => bootstrap but no cross-episode trace.
t=Trainer(device='cpu',mode='eager',n=1,horizon=3,minibatch=3,epochs=1)
t.rew.copy_(torch.tensor([[1.],[2.],[3.]]));t.val.fill_(.5);t.nextval.fill_(2)
t.term.copy_(torch.tensor([[0.],[1.],[0.]]));t.end.copy_(torch.tensor([[0.],[1.],[1.]]))
t._gae();expected=torch.tensor([[1+.99*2-.5+.99*.95*(2-.5)+.5],[2.],[3+.99*2]])
out['gae_returns_max_abs']=(t.ret-expected).abs().max().item();assert out['gae_returns_max_abs']<1e-6
out['all_passed']=True
(root/'results/validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
