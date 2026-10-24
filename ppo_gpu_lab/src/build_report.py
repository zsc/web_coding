import json,pathlib,statistics,hashlib
from webgpu_report import render as render_webgpu
root=pathlib.Path(__file__).resolve().parents[1];data=root/'results/desk'
def read(name):return json.loads((data/(name+'.json')).read_text())
js=read('js_baseline');jsms=statistics.median(x['total_ms'] for x in js['runs'][1:]);js['ms']=jsms;js['sps']=2048/(jsms/1000)
d={'js':js,'cpu':read('cpu_matched'),'cpuScaled':read('cpu_scaled'),'eager':read('eager_matched'),'eagerScaled':read('eager_scaled') if (data/'eager_scaled.json').exists() else None,'matched':read('graph_matched_fused'),'unfused':read('graph_matched'),'scaled':read('graph_scaled_seed7'),'seeds':[read('graph_scaled_seed'+str(i)) for i in [7,11,23]],'physical':read('physical_seed7'),'validation':read('validation'),'graphValidation':read('graph_validation'),'dynamics':read('dynamics_diagnostic'),'independent':json.loads((root/'results/independent_eval.json').read_text())}
d['v2']={'benchmark':json.loads((root/'results/benchmark_v2.json').read_text()),'validation':json.loads((root/'results/validation_v2.json').read_text()),'runs':[read('v2_physical_seed'+str(i)) for i in [7,11,23]]}
d['ratios']={'graph_vs_eager':d['matched']['median_sps']/d['eager']['median_sps'],'matched_vs_js':d['matched']['median_sps']/js['sps'],'scaled_vs_js':d['scaled']['median_sps']/js['sps'],'scaled_vs_cpu':d['scaled']['median_sps']/d['cpuScaled']['median_sps']}
# Each run retains its own original measurement. Report selects seed 7 a priori.
template=(root/'src/report.template.html').read_text();payload=json.dumps(d,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
live={'lqr':json.loads((root/'results/lqr_design.json').read_text()),'policy':d['v2']['runs'][0]['policy'],'evaluation':json.loads((root/'results/recovery_eval.json').read_text())}
trained=json.loads((root/'results/webgpu_train_seed71.json').read_text())
live['trained']={'policy':trained['best']['policy'],'force':40,'label':f"WebGPU PPO · seed 71 / 第 {trained['best']['iteration']} 轮"}
(root/'results/summary.json').write_text(json.dumps({**d,'interactive':live,'source_sha256':hashlib.sha256((root/'source/original.html').read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
live_payload=json.dumps(live,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
live_section=(root/'src/live_section.html').read_text()
live_style=(root/'src/live_style.css').read_text()
live_scripts='<script id="live-data" type="application/json">'+live_payload+'</script><script>'+(root/'src/live_model.js').read_text()+'</script><script>'+(root/'src/live_ui.js').read_text()+'</script>'
html=template.replace('__WEBGPU_REPORT__',render_webgpu()).replace('__REPORT_DATA__',payload).replace('__LIVE_SECTION__',live_section).replace('__LIVE_STYLE__',live_style).replace('__LIVE_SCRIPTS__',live_scripts)
html=html.replace('<a href="results/summary.json">summary.json ↗ 汇总数据</a>','<span class="small">summary.json 仅本地保留</span>')
html=html.replace('<a href="results/desk/v2_physical_seed7.pt">v2_physical_seed7.pt ↗ 新权重</a>','<span class="small">模型权重仅本地保留</span>').replace('<a href="results/desk/physical_seed7.pt">physical_seed7.pt ↗ 模型权重</a>','<span class="small">模型权重仅本地保留</span>')
(root/'report.html').write_text(html)
play='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>拽一下，看它找回平衡 · 双倒立摆实时实验</title><style>*{box-sizing:border-box}body{margin:0;background:#f2f4ee;color:#183c31;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft Yahei",sans-serif;line-height:1.65}main{max-width:1240px;margin:auto;padding:0 40px}header{display:flex;justify-content:space-between;border-bottom:1px solid #ccd5cb;padding:22px 0;font:12px ui-monospace,monospace}a{color:#246e4d}button,input,select{font:inherit}footer{font-size:11px;color:#64766c;border-top:1px solid #ccd5cb;padding:20px 0 32px}@media(max-width:700px){main{padding:0 16px}}'''+live_style+'''</style></head><body><main><header><span>FIELDNOTES / INTERACTIVE LAB</span><a href="report.html">完整 GPU 报告 ↗</a></header>'''+live_section+'''<footer>本地实时物理 · 无外部依赖 · GPU 已训练权重内嵌 · 可离线打开</footer></main>'''+live_scripts+'</body></html>'
(root/'play.html').write_text(play)
print('Wrote',root/'report.html')
