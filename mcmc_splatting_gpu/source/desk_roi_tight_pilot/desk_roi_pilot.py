import pathlib,json,time,subprocess,hashlib
out=pathlib.Path('results/roi_tight_pilot');out.mkdir(parents=True,exist_ok=True)
m={'runs':[],'complete':False,'sources':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ['src/graph_fit.cu','src/roi_fit.cu','src/desk_benchmark.cu','src/desk_roi_pilot.py']}}
for sigma in [6,5,4.5,4,3.5]:
 binary='adam_roi_'+str(sigma).replace('.','p');cmd=['./'+binary,'--mode','0','--warp-roi','1','--verify-threads','12','--steps','8000','--output',str(out/binary)]
 t=time.perf_counter();subprocess.run(cmd,capture_output=True,text=True,check=True);wall=(time.perf_counter()-t)*1000
 r=json.loads((out/(binary+'.json')).read_text());rec={k:r[k] for k in ['roi_sigma','gpu_ms','verify_ms','loss','cached_loss','max_sum_drift','pixel_mae']};rec.update(wall_ms=wall,file=binary+'.json',command=cmd,binary_sha256=hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest());m['runs'].append(rec);(out/'pilot.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(rec),flush=True)
m['complete']=True;(out/'pilot.json').write_text(json.dumps(m,indent=2)+'\n')
