import json,pathlib,re
root=pathlib.Path(__file__).resolve().parents[1]
def source(name):return (root/'src'/name).read_text()
def result(name):return json.loads((root/'results'/name).read_text())
bundle='\n'.join(re.sub(r'^import .*?;\s*$','',source(f),flags=re.M).replace('export ','') for f in ['live_model.js','webgpu_ppo.mjs','webgpu_validate.mjs','webgpu_eval.mjs','webgpu_worker.mjs'])
data={'initial':result('webgpu_initial.json'),'fixture':result('webgpu_fixture.json')}
if (root/'results/webgpu_train_seed71.json').exists():data['recorded']=result('webgpu_train_seed71.json')
live={'lqr':result('lqr_design.json'),'policy':result('desk/v2_physical_seed7.json')['policy'],'evaluation':result('recovery_eval.json')}
def payload(x):return json.dumps(x,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
base='''*{box-sizing:border-box}body{margin:0;background:#f2f4ee;color:#183c31;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft Yahei",sans-serif;line-height:1.65}main{max-width:1240px;margin:auto;padding:0 40px}header{display:flex;justify-content:space-between;border-bottom:1px solid #ccd5cb;padding:22px 0;font:12px ui-monospace,monospace}a{color:#246e4d}button,input,select{font:inherit}footer{font-size:11px;color:#64766c;border-top:1px solid #ccd5cb;padding:20px 0 32px}@media(max-width:700px){main{padding:0 16px}}'''
html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>在浏览器训练 PPO · WebGPU 双倒立摆实验</title><style>'+base+source('webgpu_style.css')+source('live_style.css')+'</style></head><body><main><header><span>FIELDNOTES / WEBGPU PPO</span><a href="report.html">完整实测报告 ↗</a></header>'+source('webgpu_section.html')+source('live_section.html')+'<footer>完整 WebGPU 训练 · 本机 GPU · 无前端外部依赖 · 训练请使用 localhost / HTTPS</footer></main>'
html+='<script id="webgpu-data" type="application/json">'+payload(data)+'</script><script id="live-data" type="application/json">'+payload(live)+'</script><script id="webgpu-worker-source" type="text/plain">'+bundle.replace('</','<\\/')+'</script>'
html+='<script>'+source('live_model.js')+'</script><script>'+source('live_ui.js')+'</script><script>'+source('webgpu_ui.js')+'</script></body></html>'
(root/'webgpu.html').write_text(html)
print('Wrote webgpu.html',len(html.encode()),'bytes')
