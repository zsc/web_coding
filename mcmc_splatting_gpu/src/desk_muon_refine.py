import json,pathlib,subprocess,time,hashlib
out=pathlib.Path('results/muon_refine');out.mkdir(parents=True,exist_ok=True)
pilot=json.loads(pathlib.Path('results/muon_pilot/pilot.json').read_text());assert pilot['complete']
base=next(r for r in pilot['runs'] if r['phase']=='calibrate' and r['mode']==0)
# Extend the grid at its lower boundary. Try zero-terminal cosine as well as the
# identical-to-Adam 5% floor. Initial timing measurements use no quality input.
m={'runs':[],'target_wall_ms':base['wall_ms'],'sources':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ['src/muon_fit.cu','src/roi_fit.cu','src/desk_benchmark.cu','src/desk_muon_refine.py']},'complete':False}
def run(mode,lr,floor,steps,phase):
 stem=f'{phase}_m{mode}_lr{lr}_f{floor}_s{steps}'
 cmd=['./muon_fit','--mode',str(mode),'--lr',str(lr),'--floor',str(floor),'--steps',str(steps),'--output',str(out/stem)]
 t=time.perf_counter();subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=120);wall=(time.perf_counter()-t)*1000
 r=json.loads((out/(stem+'.json')).read_text());rec={k:r[k] for k in ['mode','lr','steps','schedule_floor','loss','gpu_ms']};rec.update(wall_ms=wall,file=stem+'.json',phase=phase,command=cmd)
 m['runs'].append(rec);(out/'refine.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(rec),flush=True);return rec
for mode in [1,2]:
 for lr in [.03125,.0625,.125,.25]:
  r=run(mode,lr,.05,3000,'calibrate')
  overhead=r['wall_ms']-r['gpu_ms'];steps=max(32,round((base['wall_ms']-overhead)/(r['gpu_ms']/3000)/32)*32)
  for floor in [.05,0.]:run(mode,lr,floor,steps,'tune')
m['complete']=True;(out/'refine.json').write_text(json.dumps(m,indent=2)+'\n')
