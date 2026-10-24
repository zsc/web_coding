"""Additional whole-loop compiler experiment; same mathematical tensor algorithm."""
import json,time,math
from pathlib import Path
import torch
from ddpg import Agent,Runner,capture,hardware

torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
rows=[]
for n,b in [(1,64),(32,256),(256,1024)]:
    torch.manual_seed(1);a=Agent('cuda');r=Runner(a,n,b)
    for _ in range(math.ceil(5000/n)):r.collect(True)
    f=torch.compile(r.step,fullgraph=True,mode='max-autotune-no-cudagraphs')
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(5):f()
    g=capture(f);torch.cuda.synchronize();setup=time.perf_counter()-start
    times=[]
    for _ in range(5):
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(1000):g.replay()
        torch.cuda.synchronize();times.append((time.perf_counter()-start))
    median=sorted(times)[2]
    row={'device':'cuda','compile':True,'graph':True,'compile_scope':'full_loop','envs':n,'batch':b,'rounds_ms':times,'median_ms':median,'updates_per_second':1000/median,'transitions_per_second':n*1000/median,'setup_seconds':setup,'scope':'full collect + replay write/sample + minibatch Adam update + Polyak','finite':bool(torch.isfinite(r.last_loss).all().item()),'updates_per_round':1000}
    rows.append(row);print(json.dumps(row),flush=True)
    Path('results/desk_benchmark_full.json').write_text(json.dumps({'hardware':hardware(),'rows':rows},indent=2))
