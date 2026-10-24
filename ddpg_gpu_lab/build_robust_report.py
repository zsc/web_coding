from pathlib import Path
import json,re
root=Path(__file__).resolve().parent
load=lambda name:json.loads((root/'results'/name).read_text())
final=load('seeds100_td3.json')
process=load('seeds100_td3_process_time.json')
final.update(process)
# Keep every per-seed final result and all actor weights; trim redundant history lengths.
history=[{k:v for k,v in x.items() if k!='results'} for x in final['history']]
final={**final,'history':history}
prior=load('seeds100_ddpg_200k.json')
prior={k:prior[k] for k in ['config','train_seconds','history','final','heldout']}
prior['history']=[{k:v for k,v in x.items() if k!='results'} for x in prior['history']]
data={'final':final,'prior':prior,'verification':load('verification_td3.json'),'original':load('verification_original.json'),'benchmark':load('benchmark_replay.json'),'checks':load('validation_multiseed.json')+load('validation_td3.json')+load('validation_symmetry.json')}
for key,file in [('pilot','td3_pilot.json'),('symmetric_pilot','td3_symmetry_pilot.json'),('logit_pilot','td3_logit_pilot.json'),('packing','benchmark_td3.json')]:
 p=root/'results'/file
 if p.exists():
  r=json.loads(p.read_text());data[key]={k:v for k,v in r.items() if k not in ['actors','history','final','heldout']}
  if 'heldout' in r:data[key]['passed']=sum(x['survived']==x['episodes'] for x in r['heldout'])
style=re.search(r'<style>(.*?)</style>',(root/'report.template.html').read_text(),re.S).group(1)
source=(root/'robust.template.html').read_text().replace('__STYLE__',style).replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
(root/'report.html').write_text(source)
print(root/'report.html')
