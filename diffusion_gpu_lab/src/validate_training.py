"""Check graph-captured Adam against the original eager Adam for ten updates."""
import copy,json
from pathlib import Path
import torch
import torch.nn.functional as F
from fast import new_model,schedule
from core import q_sample
from train_graph import GraphTrainer

def main():
    torch.set_num_threads(4);torch.manual_seed(9)
    a=new_model().cuda();b=copy.deepcopy(a)
    g=GraphTrainer(a);opt=torch.optim.Adam(b.parameters(),lr=1e-4);p=schedule();errs=[]
    for _ in range(10):
        x=torch.randn(64,1,28,28,device='cuda');t=torch.randint(0,200,(64,),device='cuda');n=torch.randn_like(x)
        gl=g.step(x,t,n);opt.zero_grad(set_to_none=True)
        loss=F.mse_loss(b(q_sample(x,t,p,n),t.float()),n);loss.backward();opt.step()
        errs.append(abs(gl.item()-loss.item()))
    error=max((x-y).abs().max().item() for x,y in zip(a.parameters(),b.parameters()))
    r={'max_loss_abs':max(errs),'max_weight_abs_after_10_adam_steps':error,'steps':10,
       'setup_seconds':g.setup_seconds,'threshold':2e-4,'passed':error<2e-4 and max(errs)<2e-4}
    assert r['passed'],r
    Path('results/train_graph_validation.json').write_text(json.dumps(r,indent=2));print(r)

if __name__=='__main__':main()
