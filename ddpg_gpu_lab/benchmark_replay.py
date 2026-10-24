"""Diagnose full replay functionalization vs native in-place CUDA graph writes."""
import json,time,copy
from pathlib import Path
import torch
from multiseed import Packed,PackedRunner,state_tensors
from ddpg import capture

torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
rows=[]
for mode in ['full_compile','staged_compile']:
 a=Packed(range(1,101));r=PackedRunner(a,episode_limit=2000)
 for _ in range(32):r.collect(True)
 if mode=='full_compile':fn=torch.compile(r.step,fullgraph=True,mode='max-autotune-no-cudagraphs')
 else:
  prepare=torch.compile(r.prepare,fullgraph=True,mode='max-autotune-no-cudagraphs')
  learn=torch.compile(r.learn,fullgraph=True,mode='max-autotune-no-cudagraphs')
  def fn():
   index,packed=prepare();r.replay.scatter_(1,index,packed);learn()
 start=time.perf_counter();fn();torch.cuda.synchronize();g=capture(fn);torch.cuda.synchronize();setup=time.perf_counter()-start
 times=[]
 for _ in range(5):
  torch.cuda.synchronize();t=time.perf_counter()
  for _ in range(200):g.replay()
  torch.cuda.synchronize();times.append((time.perf_counter()-t)*5)
 row={'mode':mode,'rounds_ms':times,'median_ms':sorted(times)[2],'setup_seconds':setup};rows.append(row);print(json.dumps(row),flush=True)
Path('results/benchmark_replay.json').write_text(json.dumps(rows,indent=2))
