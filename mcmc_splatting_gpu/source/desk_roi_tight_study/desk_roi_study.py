import pathlib,json,time,subprocess,hashlib
root=pathlib.Path.cwd();out=root/'results/roi_tight_study';out.mkdir(parents=True,exist_ok=True)
old=json.loads((root/'results/graph_study/study.json').read_text());assert hashlib.sha256((root/'graph_fit').read_bytes()).hexdigest()==old['binaries']['graph_fit']
variants=[dict(name='original',binary='graph_fit',warp=0,threads=1,steps=8000,sigma=6),dict(name='optimized_6',binary='adam_roi_6',warp=1,threads=12,steps=8000,sigma=6),dict(name='roi_4p5',binary='adam_roi_4p5',warp=1,threads=12,steps=8000,sigma=4.5),dict(name='roi_4',binary='adam_roi_4',warp=1,threads=12,steps=8000,sigma=4),dict(name='roi_4p5_long',binary='adam_roi_4p5',warp=1,threads=12,steps=11000,sigma=4.5)]
m={'complete':False,'variants':variants,'runs':[],'baseline_sources':old['sources'],'baseline_source_archive':'source/desk_graph_study','sources':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['src/graph_fit.cu','src/roi_fit.cu','src/desk_benchmark.cu','src/desk_roi_study.py']},'binaries':{b:hashlib.sha256((root/b).read_bytes()).hexdigest() for b in {v['binary'] for v in variants}},'target_sha256':hashlib.sha256((root/'zhou_target.pgm').read_bytes()).hexdigest(),'policy':'Settings frozen from seed42 pilot. Prefer 4.5 sigma for small truncation error; also show 4 sigma tradeoff. 11000 steps preset for better quality, not described as exact equal-time. Final MSE always full, untruncated FP64.','timing':'perf_counter around full subprocess, including CPU validation and exit; no profiler, compile or SSH.'}
def hw():return subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,driver_version','--format=csv,noheader'],text=True).strip()
m['hardware_before']=hw()
def run(v,seed,phase,repeat):
 stem=f'{phase}_{repeat}_{v["name"]}_seed{seed}';cmd=['./'+v['binary'],'--mode','0','--warp-roi',str(v['warp']),'--verify-threads',str(v['threads']),'--steps',str(v['steps']),'--seed',str(seed),'--output',str(out/stem)]
 t=time.perf_counter();subprocess.run(cmd,check=True,capture_output=True,text=True);wall=(time.perf_counter()-t)*1000;r=json.loads((out/(stem+'.json')).read_text());rec={**v,'seed':seed,'phase':phase,'repeat':repeat,'wall_ms':wall,'file':stem+'.json','command':cmd,**{k:r[k] for k in ['gpu_ms','verify_ms','loss','cached_loss','max_sum_drift','pixel_mae']}}
 m['runs'].append(rec);(out/'study.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({k:v for k,v in rec.items() if k!='command'}),flush=True)
run({**variants[0],'steps':1000},42,'warmup',0)
for phase,seeds in [('repeat',[42]*3),('holdout',[1042,2042,3042])]:
 chosen=variants if phase=='repeat' else [variants[i] for i in [0,2,3,4]]
 for repeat,seed in enumerate(seeds):
  for i in range(len(chosen)):run(chosen[(i+repeat)%len(chosen)],seed,phase,repeat)
m['hardware_after']=hw();m['complete']=True;(out/'study.json').write_text(json.dumps(m,indent=2)+'\n');print('DONE',flush=True)
