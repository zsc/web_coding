"""Matched learned-update microbenchmark: 100 sequential vs packed seed learners.

Same 32/16 network, FP32, Adam, batch, initial weights and minibatch data.
Only learned updates are timed (no environment, logging or replay sampling).
"""
import argparse,json,time
from pathlib import Path
import torch
from ddpg import Agent,capture
from multiseed import Packed
p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=100);p.add_argument('--batch',type=int,default=256);p.add_argument('--iterations',type=int,default=100);p.add_argument('--out',default='results/benchmark_multiseed.json');args=p.parse_args()
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(55)
s=torch.randn(args.seeds,args.batch,4,device='cuda')*.1;act=torch.randn(args.seeds,args.batch,1,device='cuda');r=torch.rand(args.seeds,args.batch,1,device='cuda');batch=(s,act,r,s+.01,torch.zeros_like(r));rows=[]
for mode in ['sequential','packed']:
    start=time.perf_counter()
    if mode=='packed':
        agent=Packed(range(1,args.seeds+1));f=torch.compile(agent.update,fullgraph=True,mode='max-autotune-no-cudagraphs');fn=lambda:f(batch)
    else:
        agents=[]
        for seed in range(1,args.seeds+1):
            torch.manual_seed(seed);a=Agent('cpu')
            for key in ['actor','critic','target_actor','target_critic','m','v']:setattr(a,key,[t.cuda() for t in getattr(a,key)])
            a.t=a.t.cuda();agents.append(a)
        fs=[torch.compile(a.update,fullgraph=True,mode='max-autotune-no-cudagraphs') for a in agents]
        bs=[tuple(t[j].contiguous() for t in batch) for j in range(args.seeds)]
        def fn():
            for f,b in zip(fs,bs):f(b)
    fn();torch.cuda.synchronize();g=capture(fn);torch.cuda.synchronize();setup=time.perf_counter()-start;ms=[]
    for _ in range(5):
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(args.iterations):g.replay()
        torch.cuda.synchronize();ms.append((time.perf_counter()-start)*1000/args.iterations)
    row={'mode':mode,'seeds':args.seeds,'batch':args.batch,'rounds_ms_per_sweep':ms,'median_ms_per_sweep':sorted(ms)[2],'setup_seconds':setup,'scope':'one learned update for every independent seed; fixed replay batch; excludes environment/replay sampling'};rows.append(row);print(json.dumps(row),flush=True)
Path(args.out).write_text(json.dumps({'rows':rows,'speedup':rows[0]['median_ms_per_sweep']/rows[1]['median_ms_per_sweep']},indent=2))
