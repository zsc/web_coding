import json,pathlib,statistics,hashlib
root=pathlib.Path(__file__).resolve().parents[1];data=root/'results/desk'
def read(name):return json.loads((data/(name+'.json')).read_text())
js=read('js_baseline');jsms=statistics.median(x['total_ms'] for x in js['runs'][1:]);js['ms']=jsms;js['sps']=2048/(jsms/1000)
d={'js':js,'cpu':read('cpu_matched'),'cpuScaled':read('cpu_scaled'),'eager':read('eager_matched'),'eagerScaled':read('eager_scaled') if (data/'eager_scaled.json').exists() else None,'matched':read('graph_matched_fused'),'unfused':read('graph_matched'),'scaled':read('graph_scaled_seed7'),'seeds':[read('graph_scaled_seed'+str(i)) for i in [7,11,23]],'physical':read('physical_seed7'),'validation':read('validation'),'graphValidation':read('graph_validation'),'dynamics':read('dynamics_diagnostic'),'independent':json.loads((root/'results/independent_eval.json').read_text())}
d['v2']={'benchmark':json.loads((root/'results/benchmark_v2.json').read_text()),'validation':json.loads((root/'results/validation_v2.json').read_text()),'runs':[read('v2_physical_seed'+str(i)) for i in [7,11,23]]}
d['ratios']={'graph_vs_eager':d['matched']['median_sps']/d['eager']['median_sps'],'matched_vs_js':d['matched']['median_sps']/js['sps'],'scaled_vs_js':d['scaled']['median_sps']/js['sps'],'scaled_vs_cpu':d['scaled']['median_sps']/d['cpuScaled']['median_sps']}
# Each run retains its own original measurement. Report selects seed 7 a priori.
(root/'results/summary.json').write_text(json.dumps({**d,'source_sha256':hashlib.sha256((root/'source/original.html').read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
template=(root/'src/report.template.html').read_text();payload=json.dumps(d,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
(root/'report.html').write_text(template.replace('__REPORT_DATA__',payload))
print('Wrote',root/'report.html')
