"""Evaluate the saved final actor without retraining; also export a long rollout."""
import argparse,json
from pathlib import Path
import torch
from ddpg import Agent,evaluate
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--device',default='cpu');p.add_argument('--horizon',type=int,default=1000);p.add_argument('--out',default='results/saved_policy_eval.json');args=p.parse_args()
torch.set_num_threads(1)
record=json.loads(Path(args.checkpoint).read_text());a=Agent(args.device)
for w,v in zip(a.actor,record['actor']):w.copy_(torch.tensor(v,device=args.device,dtype=w.dtype))
result=evaluate(a,horizon=args.horizon)
Path(args.out).write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='trajectory'}))
