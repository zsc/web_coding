"""PyTorch CPU autograd oracle for the browser's handwritten PPO backward.
Produces exact inputs, first gradient, and twenty Adam updates. No GPU needed.
"""
import json,pathlib,torch
from ppo_v2 import actor,critic,objective
root=pathlib.Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
torch.manual_seed(991)
state=torch.load(root/'results/v2_physical_seed7.pt',map_location='cpu',weights_only=True)
if 'actor.0.weight' not in state: print(state.keys());raise ValueError('Unexpected checkpoint schema')
initial=torch.cat([v.flatten() for v in state.values()])
(root/'results/webgpu_initial.json').write_text(json.dumps(initial.tolist()))
# Validation weights are deliberately fresh: cover broad activations and PPO clipping.
w=(torch.rand(819)-.5)*.7;w[417]=-.6;w.requires_grad_()
obs=torch.randn(2048,6)*.5
raw=torch.randn(2048)*.8
with torch.no_grad():
    mu,ls=actor(w,obs)
    lp=-.5*((raw-mu)*(-ls).exp()).square()-ls-.5*torch.log(torch.tensor(2*torch.pi))
    oldlp=lp+torch.linspace(-.6,.6,2048)
adv=torch.randn(2048);ret=torch.randn(2048)*2
idx=torch.arange(2048)
out={'reference':'PyTorch 2.10 CPU FP32 autograd; ppo_v2.objective','initial':w.detach().tolist(),'obs':obs.tolist(),'raw':raw.tolist(),'oldlp':oldlp.tolist(),'adv':adv.tolist(),'ret':ret.tolist()}
# Match the two learning-rate groups with independently tracked parameters.
wa=torch.nn.Parameter(w.detach()[:418].clone());wc=torch.nn.Parameter(w.detach()[418:].clone())
optim=torch.optim.Adam([{'params':[wa],'lr':3e-4},{'params':[wc],'lr':1e-3}],foreach=False)
for i in range(20):
    optim.zero_grad();flat=torch.cat((wa,wc));loss,_=objective(flat,obs,raw,oldlp,adv,ret,idx);loss.backward()
    if i==0:out['gradient']=torch.cat((wa.grad,wc.grad)).tolist();out['loss']=float(loss.detach())
    optim.step()
    if i==0:out['after1']=torch.cat((wa,wc)).detach().tolist()
out['after20']=torch.cat((wa,wc)).detach().tolist()
(root/'results/webgpu_fixture.json').write_text(json.dumps(out,separators=(',',':')))
print('initial',len(initial),'fixture samples',len(raw),'loss',out['loss'])
