"""Describe profiler intervals; never interpret blocking runtime calls as pure launch overhead."""
import csv,json,pathlib,sqlite3
root=pathlib.Path(__file__).resolve().parents[1];folder=root/'results/desk_graph';result={'nsys':{},'ncu':{},'notes':['Profiler runs are diagnostic, not acceptance timings.','Kernel/memset interval coverage is not SM occupancy or MFU. Runtime API duration includes queue backpressure.','GPU clocks and caches left uncontrolled; other display workloads retained.']}
for name in ['adam_eager','muon_eager','adam_optimized','adam_graph32']:
 c=sqlite3.connect(folder/(name+'.sqlite'));intervals=c.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL UNION ALL SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET ORDER BY start').fetchall();start=min(a for a,b in intervals);end=max(b for a,b in intervals);covered=0;cursor=start
 for a,b in intervals:
  if b>cursor:covered+=b-max(a,cursor);cursor=b
 item={'interval_ms':(end-start)/1e6,'covered_interval_pct':100*covered/(end-start),'gaps_ms':(end-start-covered)/1e6}
 for suffix in ['cuda_api_sum','cuda_gpu_kern_sum','cuda_gpu_mem_time_sum']:
  item[suffix]=list(csv.DictReader((folder/(name+'_'+suffix+'.csv')).open()))
 result['nsys'][name]=item
for name in ['adam_ncu','optimized_ncu']:
 lines=(folder/(name+'.csv')).read_text().splitlines();start=next(i for i,s in enumerate(lines) if s.startswith('"ID"'))
 kernels={}
 for r in csv.DictReader(lines[start:]):
  k=r['Kernel Name'].split('(')[0];item=kernels.setdefault(k,{'metrics':{},'rules':[]})
  if r['Metric Name']:item['metrics'][r['Metric Name']]={'value':r['Metric Value'],'unit':r['Metric Unit']}
  if r.get('Rule Description'):item['rules'].append(r['Rule Description'])
 result['ncu'][name]=kernels
result['versions']=(folder/'versions.txt').read_text();(folder/'profile_summary.json').write_text(json.dumps(result,indent=2)+'\n');print('Saved profiler summary.')
