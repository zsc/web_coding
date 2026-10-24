"""Fresh paired-seed experiment after tuning; freeze wall-time calibrated budgets."""
import json,pathlib,subprocess,time,hashlib
root=pathlib.Path.cwd();out=root/'results/muon_study';out.mkdir(parents=True,exist_ok=True)
pilot=json.loads((root/'results/muon_pilot/pilot.json').read_text());refine=json.loads((root/'results/muon_refine/refine.json').read_text());assert pilot['complete'] and refine['complete']
tuned_adam=min([r for r in pilot['runs'] if r['phase']=='tune' and r['mode']==0],key=lambda r:r['loss'])
choices={mode:min([r for r in refine['runs'] if r['phase']=='tune' and r['mode']==mode],key=lambda r:r['loss']) for mode in [1,2]}
variants=[dict(name='adam_default',mode=0,lr=1.,floor=.05,steps=8000),dict(name='adam_tuned',mode=0,lr=tuned_adam['lr'],floor=.05,steps=8000)]+[dict(name=name,mode=mode,lr=choices[mode]['lr'],floor=choices[mode]['schedule_floor'],steps=choices[mode]['steps']) for mode,name in [(1,'muon_joint'),(2,'muon_grouped')]]
manifest={'complete':False,'runs':[],'tuning_seed_base':42,'holdout_seed_bases':[1042,2042,3042],
 'selection':{'adam':tuned_adam,'muon':choices,'rule':'Minimum seed42 final original quantized MSE from the recorded grids. Muon lower-grid extension adds 0-terminal cosine after the first optimum was on the low-LR boundary. No holdout outcome used.'},
 'calibration_rule':'Measure each selected setting once. Take fresh default Adam total wall time as target. Estimate overhead = full wall minus GPU time, per-step cost = GPU time / steps. Rescale remaining steps to target, nearest 32. Freeze before all repeat and holdout runs.',
 'sources':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['src/muon_fit.cu','src/roi_fit.cu','src/desk_benchmark.cu','src/desk_muon_study.py']},
 'target_sha256':hashlib.sha256((root/'zhou_target.pgm').read_bytes()).hexdigest(),
 'timing':'perf_counter around subprocess.run, includes launch/CUDA context/init/optimization/readback/full FP64 renderer/output/cleanup/exit, excludes compile/SSH/harness setup.'}
manifest['hardware_before']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader'],text=True).strip()
def run(v,seed,phase,repeat):
 stem=f'{phase}_{repeat}_{v["name"]}_seed{seed}'
 cmd=['./muon_fit','--input','zhou_target.pgm','--gaussians','1600','--chains','8','--mode',str(v['mode']),'--lr',str(v['lr']),'--floor',str(v['floor']),'--steps',str(v['steps']),'--seed',str(seed),'--output',str(out/stem)]
 t=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=120);wall=(time.perf_counter()-t)*1000
 r=json.loads((out/(stem+'.json')).read_text());rec={**v,'group':v['name'],'phase':phase,'repeat':repeat,'seed':seed,'file':stem+'.json','wall_ms':wall,'gpu_ms':r['gpu_ms'],'loss':r['loss'],'smooth_loss':r['smooth_loss'],'command':cmd}
 manifest['runs'].append(rec);(out/'study.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in rec.items() if k!='command'}),flush=True);return rec
run({**variants[0],'steps':1000},42,'warmup',0)
base=run(variants[0],42,'calibrate',0);calibration=[base]
for v in variants[1:]:
 r=run(v,42,'calibrate',0);calibration.append(r)
 v['steps']=max(32,round((base['wall_ms']-(r['wall_ms']-r['gpu_ms']))/(r['gpu_ms']/v['steps'])/32)*32)
manifest['variants']=[dict(v) for v in variants];manifest['target_wall_ms']=base['wall_ms']
for phase,seeds in [('repeat',[42]*3),('holdout',[1042,2042,3042])]:
 for repeat,seed in enumerate(seeds):
  for i in range(4):run(variants[(i+repeat)%4],seed,phase,repeat)
manifest['hardware_after']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader'],text=True).strip()
manifest['complete']=True;(out/'study.json').write_text(json.dumps(manifest,indent=2)+'\n');print('DONE',flush=True)
