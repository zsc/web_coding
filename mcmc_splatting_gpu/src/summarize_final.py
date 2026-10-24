#!/usr/bin/env python3
import json,pathlib,statistics,csv,hashlib,math
root=pathlib.Path(__file__).resolve().parents[1]
runs=[]
for i in range(1,4):
 r=json.loads((root/f'results/desk/final_repeat_{i}.json').read_text())
 r['process_wall_seconds']=float((root/f'results/desk/final_repeat_{i}.time').read_text().split('=')[1]);r['file']=f'final_repeat_{i}';runs.append(r)
representative=sorted(runs,key=lambda r:r['process_wall_seconds'])[1]
s=(root/'results/desk/ncu_final.csv').read_text();rows=list(csv.DictReader(s[s.index('"ID"'):].splitlines()));kernels={}
for row in rows:
 if row.get('Metric Name'):
  k=kernels.setdefault(row['ID'],{'name':row['Kernel Name'].split('(')[0],'metrics':{}})
  k['metrics'][row['Metric Name']]=float(row['Metric Value'].replace(',',''))
for k in kernels.values():
 m=k['metrics'];flops=sum(m['smsp__sass_thread_inst_executed_op_'+n+'_pred_on.sum']*factor for n,factor in [('fadd',1),('fmul',1),('ffma',2)])
 seconds=m['gpu__time_duration.sum']*1e-9;peak=48*64*2*m['sm__cycles_elapsed.avg.per_second']
 k.update(flops=flops,seconds=seconds,peak_flops_s=peak,arithmetic_utilization_pct=100*flops/seconds/peak)
pair=100*sum(k['flops'] for k in kernels.values())/sum(k['seconds']*k['peak_flops_s'] for k in kernels.values())
web=json.loads((root/'results/webgpu_precision.json').read_text());wm={}
for mode in ['fp32','f16-cache','f16-exp']:
 rs=[r for r in web if r['mode']==mode];wm[mode]={'median_gpu_ms':statistics.median(r['gpu_ms'] for r in rs),'median_wall_ms':statistics.median(r['wall_ms'] for r in rs),'rerender_loss':rs[0]['rerender_loss'],'render_mae':rs[0]['render_mae'],'cache_MiB':rs[0]['cache_bytes']/1048576}
out={'acceptance_passed':all(r['process_wall_seconds']<10 and r['loss']<95.8913818359 and r['sse_invariant'] for r in runs),'runs':runs,'representative':representative,'process_median_s':statistics.median(r['process_wall_seconds'] for r in runs),'process_max_s':max(r['process_wall_seconds'] for r in runs),'mse_range':[min(r['loss'] for r in runs),max(r['loss'] for r in runs)],'profile_kernels':kernels,'profile_pair_arithmetic_utilization_pct':pair,'webgpu':wm,'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['src/roi_fit.cu','src/webgpu.mjs']}}
(root/'results/final_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({k:out[k] for k in ['acceptance_passed','process_median_s','process_max_s','mse_range','profile_pair_arithmetic_utilization_pct','webgpu']},ensure_ascii=False,indent=2))

