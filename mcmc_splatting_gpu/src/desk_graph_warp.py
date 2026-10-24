import json,pathlib,subprocess,time,hashlib
out=pathlib.Path('results/graph_warp');out.mkdir(parents=True,exist_ok=True)
m={'runs':[],'complete':False,'sources':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ['src/graph_fit.cu','src/roi_fit.cu','src/desk_benchmark.cu','src/desk_graph_warp.py']}}
for mode,steps,N,C in [(0,8,32,2),(1,8,32,2),(2,8,32,2),(0,8000,1600,8),(1,6912,1600,8)]:
 for warp,batch in ([(0,0),(1,0)] if steps==8 else [(0,0),(1,0),(1,32)]):
  stem=f'm{mode}_w{warp}_b{batch}_s{steps}'
  cmd=['./graph_fit_fma','--mode',str(mode),'--lr',str(1 if mode==0 else .25),'--warp-roi',str(warp),'--graph-batch',str(batch),'--verify-threads','12','--steps',str(steps),'--gaussians',str(N),'--chains',str(C),'--output',str(out/stem)]
  t=time.perf_counter();subprocess.run(cmd,check=True,capture_output=True,text=True);wall=(time.perf_counter()-t)*1000
  r=json.loads((out/(stem+'.json')).read_text());rec={k:r[k] for k in ['mode','steps','warp_roi','graph_batch','graph_build_ms','verify_threads','verify_ms','gpu_ms','loss']};rec.update(wall_ms=wall,file=stem+'.json',command=cmd);m['runs'].append(rec);(out/'pilot.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(rec),flush=True)
m['complete']=True;(out/'pilot.json').write_text(json.dumps(m,indent=2)+'\n')
