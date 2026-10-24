import json,pathlib,subprocess,time,hashlib
out=pathlib.Path('results/muon_pilot');out.mkdir(parents=True,exist_ok=True)
m={'runs':[],'sources':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ['src/muon_fit.cu','src/roi_fit.cu','src/desk_benchmark.cu','src/desk_muon_pilot.py']},'complete':False}
def run(mode,lr,steps,phase,N=1600,C=8,audit=False):
 stem=f'{phase}_m{mode}_lr{lr}_s{steps}_n{N}'
 cmd=['./muon_fit','--input','zhou_target.pgm','--mode',str(mode),'--lr',str(lr),'--steps',str(steps),'--gaussians',str(N),'--chains',str(C),'--output',str(out/stem)]
 if audit:cmd+=['--audit',str(out/(stem+'_audit.json'))]
 t=time.perf_counter();p=subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=120);wall=(time.perf_counter()-t)*1000
 r=json.loads((out/(stem+'.json')).read_text());rec={k:r[k] for k in ['mode','lr','steps','N','chains','loss','gpu_ms','setup_ms']};rec.update(wall_ms=wall,file=stem+'.json',phase=phase,command=cmd)
 m['runs'].append(rec);(out/'pilot.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(rec),flush=True);return rec
for mode in [1,2]:
 run(mode,1,1,'audit',N=32,C=2,audit=True)
 run(mode,1,3,'audit',N=1600,C=2,audit=True)
run(0,1,1000,'warmup')
base=run(0,1,8000,'calibrate')
budgets={0:8000}
for mode in [1,2]:
 r=run(mode,1,4000,'calibrate')
 overhead=r['wall_ms']-r['gpu_ms'];steps=round((base['wall_ms']-overhead)/(r['gpu_ms']/4000)/32)*32
 budgets[mode]=max(32,steps)
m['budgets']=budgets
for mode,grid in [(0,[.5,1.,2.,4.]),(1,[.25,.5,1.,2.,4.]),(2,[.25,.5,1.,2.,4.])]:
 for lr in grid:run(mode,lr,budgets[mode],'tune')
m['complete']=True;(out/'pilot.json').write_text(json.dumps(m,indent=2)+'\n')
