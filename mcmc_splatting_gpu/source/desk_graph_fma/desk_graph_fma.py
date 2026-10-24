import json,pathlib,subprocess,time,hashlib
out=pathlib.Path('results/graph_fma');out.mkdir(parents=True,exist_ok=True)
m={'runs':[],'complete':False,'sources':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ['src/graph_fit.cu','src/roi_fit.cu','src/desk_benchmark.cu','src/desk_graph_fma.py']},'binaries':{b:hashlib.sha256(pathlib.Path(b).read_bytes()).hexdigest() for b in ['graph_fit','graph_fit_fma']}}
for mode,steps in [(0,8000),(1,6912)]:
 for binary,batch,threads in [('graph_fit',0,1),('graph_fit_fma',0,1),('graph_fit_fma',1,1),('graph_fit_fma',32,1),('graph_fit_fma',0,12),('graph_fit_fma',1,12)]:
  stem=f'{binary}_m{mode}_b{batch}_t{threads}'
  cmd=['./'+binary,'--mode',str(mode),'--lr',str(1 if mode==0 else .25),'--graph-batch',str(batch),'--verify-threads',str(threads),'--steps',str(steps),'--output',str(out/stem)]
  t=time.perf_counter();subprocess.run(cmd,check=True,capture_output=True,text=True);wall=(time.perf_counter()-t)*1000
  r=json.loads((out/(stem+'.json')).read_text());rec={k:r[k] for k in ['mode','steps','graph_batch','graph_build_ms','verify_threads','verify_ms','gpu_ms','loss']};rec.update(binary=binary,wall_ms=wall,file=stem+'.json',command=cmd);m['runs'].append(rec);(out/'pilot.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(rec),flush=True)
m['complete']=True;(out/'pilot.json').write_text(json.dumps(m,indent=2)+'\n')
