import torch,json,time,pathlib,statistics,subprocess
from ppo import Trainer
from ppo_v2 import TrainerV2
root=pathlib.Path(__file__).resolve().parents[1];torch.set_num_threads(1)
setup={};runs={};models={}
# Ablations keep N/T/minibatch/epochs identical. Run sequentially, never concurrently.
for name,cls,opts in [('v1',Trainer,{}),('flat_adam',TrainerV2,{'rollout_v2':False}),('bulk_critic',TrainerV2,{'optimizer_v2':False}),('v2',TrainerV2,{})]:
 start=time.perf_counter();t=cls(**opts);t.prepare();torch.cuda.synchronize();setup[name]=time.perf_counter()-start;models[name]=t;runs[name]=[]
 print(json.dumps({'prepared':name,'seconds':setup[name]}),flush=True)
for i in range(30):
 names=list(models) if i%2==0 else list(reversed(models))
 for name in names:
  t=models[name];torch.cuda.synchronize();start=time.perf_counter();t.step();torch.cuda.synchronize();elapsed=time.perf_counter()-start
  assert torch.isfinite(t.metrics).all()
  runs[name].append(elapsed)
 if (i+1)%10==0:print(json.dumps({'pair':i+1,'ms':{k:v[-1]*1e3 for k,v in runs.items()}}),flush=True)
result={'config':{'environments':256,'horizon':64,'minibatch':2048,'epochs':10,'dtype':'float32','parameters':819},'protocol':'Four variants interleaved in alternating forward/reverse order; 30 samples each; CUDA synchronized; no simultaneous model execution. Shared GPU background load remains.','setup_seconds':setup,'measurements':{k:{'median_ms':statistics.median(v)*1000,'median_sps':16384/statistics.median(v),'p10_ms':sorted(v)[3]*1000,'p90_ms':sorted(v)[26]*1000,'seconds':v} for k,v in runs.items()}}
result['speedup_v1_to_v2']=result['measurements']['v1']['median_ms']/result['measurements']['v2']['median_ms']
result['speedup_each_pair']=[a/b for a,b in zip(runs['v1'],runs['v2'])]
(root/'results/benchmark_v2.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
