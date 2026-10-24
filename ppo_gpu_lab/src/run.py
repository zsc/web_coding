import argparse,json,time,pathlib,platform,subprocess,statistics,torch
from ppo import Trainer
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['cpu','eager','graph','v2'],default='graph');p.add_argument('--envs',type=int,default=256);p.add_argument('--horizon',type=int,default=64);p.add_argument('--minibatch',type=int,default=2048);p.add_argument('--epochs',type=int,default=10);p.add_argument('--iterations',type=int,default=20);p.add_argument('--seed',type=int,default=7);p.add_argument('--physical',action='store_true');p.add_argument('--evaluate',action='store_true');p.add_argument('--out',default='run');args=p.parse_args()
torch.set_num_threads(1)
if torch.cuda.is_available():torch.backends.cuda.matmul.allow_tf32=False
device='cpu' if args.mode=='cpu' else 'cuda'
def sync():
    if device=='cuda':torch.cuda.synchronize()
if args.mode=='v2':
    from ppo_v2 import TrainerV2
    trainer_class=TrainerV2
else:trainer_class=Trainer
start=time.perf_counter();trainer=trainer_class(device,'graph' if args.mode in ('graph','v2') else 'eager',args.envs,args.horizon,args.minibatch,args.epochs,args.seed,args.physical)
initial_eval_start=time.perf_counter()
initial=trainer.evaluate(kind='policy') if args.evaluate else None
zero=trainer.evaluate(kind='zero') if args.evaluate else None
initial_eval_seconds=time.perf_counter()-initial_eval_start
trainer.prepare();sync();setup=time.perf_counter()-start-initial_eval_seconds
print(json.dumps({'event':'prepared','setup_s':setup}),flush=True)
if device=='cuda':torch.cuda.reset_peak_memory_stats()
records=[];start=time.perf_counter()
for i in range(args.iterations):
    sync();t=time.perf_counter();trainer.step();sync();elapsed=time.perf_counter()-t
    assert torch.isfinite(trainer.metrics).all(), 'Non-finite loss'
    record={'iteration':i+1,'seconds':elapsed,'sps':trainer.b/elapsed,'wall_s':time.perf_counter()-start,'samples':(i+1)*trainer.b,'metrics':trainer.metrics.cpu().tolist()}
    if args.evaluate and (i==0 or (i+1)%20==0 or i==args.iterations-1):
        record['eval']=trainer.evaluate(count=64,seed=2026)
        record['eval'].pop('returns');record['eval'].pop('lengths');record['eval'].pop('trajectory')
    records.append(record)
    if i==0 or (i+1)%10==0:print(json.dumps(record),flush=True)
result={'config':vars(args),'host':platform.node(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'setup_seconds':setup,'initial_eval_seconds':initial_eval_seconds,'warmup_iterations_discarded':2 if args.mode in ('graph','v2') else 1,'records':records,'median_sps':trainer.b/statistics.median(x['seconds'] for x in records),'median_iteration_seconds':statistics.median(x['seconds'] for x in records),'measured_training_seconds':sum(x['seconds'] for x in records),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else 0,'initial':initial,'zero_action':zero}
if args.evaluate:
    result['final']=trainer.evaluate(seed=123,count=128,trajectory=True)
    # A second held-out reset batch assesses seed sensitivity without retraining.
    result['held_out']=trainer.evaluate(seed=999,count=128)
    result['policy']={k:v.detach().cpu().tolist() for k,v in trainer.agent.actor.state_dict().items()}
root=pathlib.Path(__file__).resolve().parents[1];(root/'results').mkdir(exist_ok=True)
(root/'results'/f'{args.out}.json').write_text(json.dumps(result,indent=2));torch.save(trainer.agent.state_dict(),root/'results'/f'{args.out}.pt')
print(json.dumps({'saved':args.out,'median_sps':result['median_sps'],'peak_allocated_mib':result['peak_allocated_mib']}),flush=True)
