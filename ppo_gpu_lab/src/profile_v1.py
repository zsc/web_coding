import torch,time,json,pathlib
from ppo import Trainer
from torch.profiler import profile,ProfilerActivity
root=pathlib.Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
t=Trainer();t.prepare()
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
 t.step();torch.cuda.synchronize()
print(p.key_averages().table(sort_by='self_cuda_time_total',row_limit=25))
p.export_chrome_trace(str(root/'results/profile_v1.json'))
# Graph-captured phase timings, preserving the phase's actual GPU work.
def update():
 obs=t.obs.reshape(-1,6);raw=t.raw.flatten();lp=t.lp.flatten();adv=t.adv.flatten();ret=t.ret.flatten()
 for _ in range(t.epochs):
  idx=torch.randperm(t.b,device=t.device)
  for start in range(0,t.b,t.mb):
   j=idx[start:start+t.mb];t.optimizer.zero_grad(set_to_none=True)
   loss,m=t.loss_fn(obs[j],raw[j],lp[j],adv[j],ret[j]);loss.backward();t.optimizer.step();t.metrics.copy_(m)
results={}
for name,fn in [('collect',t.collect),('gae',t.gae_fn),('update',update)]:
 s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(s):
  fn();fn()
 torch.cuda.current_stream().wait_stream(s);torch.cuda.synchronize()
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):fn()
 vals=[]
 for i in range(15):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
  a.record();g.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b))
 results[name]={'median_ms':sorted(vals)[len(vals)//2],'runs_ms':vals}
print(json.dumps(results,indent=2));(root/'results/phase_profile_v1.json').write_text(json.dumps(results,indent=2))
