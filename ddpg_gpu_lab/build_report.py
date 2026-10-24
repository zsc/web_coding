from pathlib import Path
import json
root=Path(__file__).resolve().parent
def read(name,default=None):
    p=root/'results'/name
    return json.loads(p.read_text()) if p.exists() else default
data={'baseline':read('desk_baseline.json'),'benchmark':read('desk_benchmark.json'),'full':read('desk_benchmark_full.json'),'validation':read('validation.json'),'training':[json.loads(p.read_text()) for p in sorted((root/'results').glob('train_gpu_seed*.json'))],'original_train':read('train_original_seed1.json'),'extended':[json.loads(p.read_text()) for p in sorted((root/'results').glob('train_extended_seed*.json'))]}
source=(root/'report.template.html').read_text()
(root/'report.html').write_text(source.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')))
print(root/'report.html')
