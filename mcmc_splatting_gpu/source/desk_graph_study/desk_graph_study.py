"""Matched-work ablations, paired holdouts, and time-only extra-step calibration."""
import hashlib,json,pathlib,subprocess,time
root=pathlib.Path.cwd();out=root/'results/graph_study';out.mkdir(parents=True,exist_ok=True)
def variant(name,mode=0,fma=False,graph=0,warp=0,threads=1,steps=8000):return dict(name=name,mode=mode,binary='graph_fit_fma' if fma else 'graph_fit',graph=graph,warp=warp,threads=threads,steps=steps,lr=1 if mode==0 else .25)
variants=[variant('adam_eager'),variant('adam_graph',graph=32),variant('adam_fma',fma=True),variant('adam_parallel',fma=True,threads=12),variant('adam_warp',fma=True,warp=1,threads=12),variant('adam_warp_graph',fma=True,warp=1,graph=32,threads=12),variant('muon_eager',mode=1,steps=6912),variant('muon_optimized',mode=1,fma=True,threads=12,steps=6912),variant('muon_graph',mode=1,fma=True,threads=12,graph=32,steps=6912)]
m={'complete':False,'runs':[],'variants':variants,'sources':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['src/graph_fit.cu','src/roi_fit.cu','src/desk_benchmark.cu','src/desk_graph_study.py']},'binaries':{b:hashlib.sha256((root/b).read_bytes()).hexdigest() for b in ['graph_fit','graph_fit_fma']},'compile_flags':'nvcc -O3 -std=c++17 -arch=sm_75 -Xcompiler -fopenmp; graph_fit --fmad=false, graph_fit_fma --fmad=true','target_sha256':hashlib.sha256((root/'zhou_target.pgm').read_bytes()).hexdigest(),'timing':'perf_counter around whole subprocess; includes graph creation, upload, FP64 validation, cleanup and process exit; excludes compile, SSH and profiling. GPU events exclude graph creation and CPU validation; each separately measured.'}
def hardware():return subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,driver_version','--format=csv,noheader'],text=True).strip()
m['hardware_before']=hardware()
def run(v,seed,phase,repeat):
 stem=f'{phase}_{repeat}_{v["name"]}_seed{seed}'
 cmd=['./'+v['binary'],'--mode',str(v['mode']),'--lr',str(v['lr']),'--warp-roi',str(v['warp']),'--graph-batch',str(v['graph']),'--verify-threads',str(v['threads']),'--steps',str(v['steps']),'--seed',str(seed),'--output',str(out/stem)]
 t=time.perf_counter();subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=120);wall=(time.perf_counter()-t)*1000
 r=json.loads((out/(stem+'.json')).read_text());rec={**v,'phase':phase,'repeat':repeat,'seed':seed,'file':stem+'.json','wall_ms':wall,'gpu_ms':r['gpu_ms'],'verify_ms':r['verify_ms'],'graph_build_ms':r['graph_build_ms'],'loss':r['loss'],'command':cmd}
 m['runs'].append(rec);(out/'study.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({k:v for k,v in rec.items() if k!='command'}),flush=True);return rec
run({**variants[0],'steps':1000},42,'warmup',0)
base=run(variants[0],42,'calibrate',0)
a=run(variants[4],42,'calibrate',0);b=run({**variants[4],'steps':12000},42,'calibrate',1)
slope=(b['wall_ms']-a['wall_ms'])/4000;assert slope>0
extra=variant('adam_equal_time',fma=True,warp=1,threads=12,steps=max(8000,min(20000,round((8000+(base['wall_ms']-a['wall_ms'])/slope)/32)*32)))
m['extra_variant']=extra;m['equal_time_calibration']={'target_ms':base['wall_ms'],'optimized_8000_ms':a['wall_ms'],'optimized_12000_ms':b['wall_ms'],'ms_per_step':slope,'rule':'Linear interpolation/extrapolation of total wall time using optimized 8000 and 12000 steps, nearest 32, bounded 8000..20000. No quality input; frozen before repetition and holdouts.'}
repeat_variants=variants+[extra]
for repeat in range(3):
 for i in range(len(repeat_variants)):run(repeat_variants[(i+repeat*3)%len(repeat_variants)],42,'repeat',repeat)
held=[variants[0],variants[4],variants[6],variants[7],extra]
for repeat,seed in enumerate([1042,2042,3042]):
 for i in range(len(held)):run(held[(i+repeat*2)%len(held)],seed,'holdout',repeat)
m['hardware_after']=hardware();m['complete']=True;(out/'study.json').write_text(json.dumps(m,indent=2)+'\n');print('DONE',flush=True)
