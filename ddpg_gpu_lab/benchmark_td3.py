"""Matched TD3 learned-update microbenchmark, serial vs packed independent seeds.
Same weights, data, FP32, TD3 symmetry, Adam and CUDA Graph in both arms.
Environment, replay sampling, initialization and compile are excluded.
"""
import argparse,json,time,gc
from pathlib import Path
import torch
from ddpg import capture
from multiseed import PackedTD3
p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=100);p.add_argument('--batch',type=int,default=128);p.add_argument('--iterations',type=int,default=100);p.add_argument('--out',default='results/benchmark_td3.json');args=p.parse_args()
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(55)
s=torch.randn(args.seeds,args.batch,4,device='cuda')*.1;act=torch.randn(args.seeds,args.batch,1,device='cuda');r=torch.rand(args.seeds,args.batch,1,device='cuda');batch=(s,act,r,s+.01,torch.zeros_like(r));rows=[]
# Each arm runs in its own function, releasing graphs before the next arm.
def measure(mode):
 start=time.perf_counter()
 if mode=='packed':
  agents=[PackedTD3(range(1,args.seeds+1),symmetry=True,lr_actor=.0003,lr_critic=.0003)];bs=[batch]
 else:
  agents=[PackedTD3([seed],symmetry=True,lr_actor=.0003,lr_critic=.0003) for seed in range(1,args.seeds+1)];bs=[tuple(t[j:j+1].contiguous() for t in batch) for j in range(args.seeds)]
 funcs=[torch.compile(a.update,fullgraph=True,mode='max-autotune-no-cudagraphs') for a in agents]
 def fn():
  for f,b in zip(funcs,bs):f(b)
 fn();torch.cuda.synchronize();g=capture(fn);torch.cuda.synchronize();setup=time.perf_counter()-start;ms=[]
 for _ in range(5):
  torch.cuda.synchronize();start=time.perf_counter()
  for _ in range(args.iterations):g.replay()
  torch.cuda.synchronize();ms.append((time.perf_counter()-start)*1000/args.iterations)
 return {'mode':mode,'seeds':args.seeds,'batch':args.batch,'rounds_ms_per_sweep':ms,'median_ms_per_sweep':sorted(ms)[2],'setup_seconds':setup}
for mode in ['sequential','packed']:
 row=measure(mode);rows.append(row);print(json.dumps(row),flush=True);gc.collect();torch.cuda.empty_cache()
Path(args.out).write_text(json.dumps({'scope':'TD3 symmetric learned updates only; same fixed minibatch and starting weights; no environment/replay sampling; five 100-sweep rounds','rows':rows,'speedup':rows[0]['median_ms_per_sweep']/rows[1]['median_ms_per_sweep']},indent=2))
