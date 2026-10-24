import json,pathlib,subprocess,time,hashlib
out=pathlib.Path('results/graph_pilot');out.mkdir(parents=True,exist_ok=True)
m={'runs':[],'complete':False,'sources':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ['src/graph_fit.cu','src/roi_fit.cu','src/desk_benchmark.cu','src/desk_graph_pilot.py']}}
def run(mode,batch,threads,steps=8000,N=1600,C=8,phase='pilot'):
 lr=1 if mode==0 else .25
 stem=f'{phase}_m{mode}_b{batch}_t{threads}_s{steps}_n{N}'
 cmd=['./graph_fit','--mode',str(mode),'--lr',str(lr),'--graph-batch',str(batch),'--verify-threads',str(threads),'--steps',str(steps),'--gaussians',str(N),'--chains',str(C),'--output',str(out/stem)]
 t=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True,check=True);wall=(time.perf_counter()-t)*1000
 r=json.loads((out/(stem+'.json')).read_text());rec={k:r[k] for k in ['mode','steps','N','chains','graph_batch','graph_build_ms','graph_iterations','verify_threads','verify_ms','gpu_ms','loss']};rec.update(wall_ms=wall,file=stem+'.json',phase=phase,command=cmd);m['runs'].append(rec);(out/'pilot.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(rec),flush=True)
for mode in [0,1,2]:
 for batch in [0,3]:run(mode,batch,1,steps=8,N=32,C=2,phase='check')
for mode in [0,1]:
 for batch in [0,1,32,128,512]:run(mode,batch,1,steps=8000 if mode==0 else 6912)
for threads in [6,12]:run(0,128,threads)
m['complete']=True;(out/'pilot.json').write_text(json.dumps(m,indent=2)+'\n')
