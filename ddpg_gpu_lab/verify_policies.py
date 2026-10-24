"""Fresh initial states, saved actors only; no gradient or replay is loaded."""
import argparse,json,time
from pathlib import Path
import torch
from multiseed import Packed,PackedTD3,EvalEngine,fw
p=argparse.ArgumentParser();p.add_argument('summary');p.add_argument('--horizon',type=int,default=100000);p.add_argument('--episodes',type=int,default=256);p.add_argument('--split',type=int,default=997);p.add_argument('--out',default='results/verification_td3.json');args=p.parse_args()
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
start=time.perf_counter()
s=json.loads(Path(args.summary).read_text());seeds=list(map(int,s['actors']));cls=PackedTD3 if s['config']['algo']=='td3' else Packed
# Historical post-tanh exports retain their original inference semantics.
if s['config'].get('symmetry') and s['config'].get('symmetry_mode')!='pre_tanh_logit_odd':
 class LegacyPostTanh(PackedTD3):
  def action(self,x):return .5*(fw(self.actor,x*self.obs_scale,True)[0]-fw(self.actor,-x*self.obs_scale,True)[0])
 cls=LegacyPostTanh
a=cls(seeds,**({'symmetry':s['config'].get('symmetry',False)} if s['config']['algo']=='td3' else {}))
for j,seed in enumerate(seeds):
 for i,w in enumerate(a.actor):w[j].copy_(torch.tensor(s['actors'][str(seed)][i],device='cuda').reshape_as(w[j]))
e=EvalEngine(a,args.episodes);setup=time.perf_counter()-start
rows,seconds=e.run(args.horizon,args.split)
result={'config':vars(args),'setup_seconds':setup,'eval_seconds':seconds,'passed':sum(x['survived']==args.episodes for x in rows),'seeds':len(seeds),'results':rows}
Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='results'}))
