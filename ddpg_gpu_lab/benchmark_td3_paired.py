"""Interleaved same-algorithm TD3 benchmark: 100 serial vs packed updates."""
import json,time
from pathlib import Path
import torch
from ddpg import capture
from multiseed import PackedTD3

torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(55)
s=torch.randn(100,128,4,device='cuda')*.1;act=torch.randn(100,128,1,device='cuda');r=torch.rand(100,128,1,device='cuda');batch=(s,act,r,s+.01,torch.zeros_like(r))
def build(mode):
 start=time.perf_counter()
 if mode=='packed':
  agents=[PackedTD3(range(1,101),symmetry=True,lr_actor=.0003,lr_critic=.0003)];bs=[batch]
 else:
  agents=[PackedTD3([seed],symmetry=True,lr_actor=.0003,lr_critic=.0003) for seed in range(1,101)];bs=[tuple(t[j:j+1].contiguous() for t in batch) for j in range(100)]
 fs=[torch.compile(a.update,fullgraph=True,mode='max-autotune-no-cudagraphs') for a in agents]
 def fn():
  for f,b in zip(fs,bs):f(b)
 fn();torch.cuda.synchronize();g=capture(fn);torch.cuda.synchronize()
 return {'agents':agents,'fns':fs,'graph':g,'setup_seconds':time.perf_counter()-start,'times':[]}
models={m:build(m) for m in ['sequential','packed']};order=[]
for repeat in range(5):
 modes=['sequential','packed'] if repeat%2==0 else ['packed','sequential'];order.append(modes)
 for mode in modes:
  model=models[mode];torch.cuda.synchronize();start=time.perf_counter()
  for _ in range(100):model['graph'].replay()
  torch.cuda.synchronize();ms=(time.perf_counter()-start)*10;model['times'].append(ms)
  print(json.dumps({'round':repeat,'mode':mode,'ms_per_sweep':ms}),flush=True)
rows=[{'mode':mode,'seeds':100,'batch':128,'rounds_ms_per_sweep':m['times'],'median_ms_per_sweep':sorted(m['times'])[2],'setup_seconds':m['setup_seconds']} for mode,m in models.items()]
Path('results/benchmark_td3.json').write_text(json.dumps({'scope':'Interleaved TD3 logit-symmetric learned updates only; fixed minibatch and starting weights; excludes environment/replay sampling; 100 sweeps per round','order':order,'rows':rows,'speedup':rows[0]['median_ms_per_sweep']/rows[1]['median_ms_per_sweep']},indent=2))
