"""Compare respaced stochastic sampling to deterministic DDIM, same checkpoint."""
import gc,json,time
from pathlib import Path
import torch
from fast import new_model
from quality_v2 import QualitySampler
from benchmark import save_grid,summary,encode_images,status

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    out=Path('results/quality_v2');ckpt=torch.load(out/'checkpoint.pt',weights_only=True)
    m=new_model().cuda().eval();m.load_state_dict(ckpt['ema_state_dict'])
    result=json.loads((out/'evaluation.json').read_text())
    result['timings']=[r for r in result['timings'] if not r['name'].endswith('_stochastic')]
    for steps in [100,50]:
        name=f'ema{steps}_stochastic';s=QualitySampler(m,steps=steps,eta=1.)
        images=[s.run(seed=seed) for seed in [881,882,883,884]]
        save_grid(images[0],out/f'{name}_25.png')
        save_grid(torch.cat(images),out/f'{name}_100.png',columns=10,scale=3)
        del s;gc.collect();s=QualitySampler(m,batch=5,steps=steps,eta=1.);s.run()
        values=[]
        for _ in range(7):
            torch.cuda.synchronize();t=time.perf_counter();encode_images(s.run());torch.cuda.synchronize()
            values.append((time.perf_counter()-t)*1000)
        result['timings'].append({'name':name,'steps':steps,'clip':True,'method':'ddim','eta':1.,**summary(values),'setup_seconds':s.setup_seconds})
        print(name,result['timings'][-1]['median_ms'],flush=True);del s;gc.collect()
    graph=QualitySampler(m,batch=5,steps=50,eta=1.,amp=False)
    eager=QualitySampler(m,batch=5,steps=50,eta=1.,amp=False,graph=False)
    a=graph.run(seed=91);b=eager.run(seed=91);diff=(a-b).abs().max().item()
    assert diff<2e-4,diff
    result['stochastic_graph_vs_eager_max_abs']=diff;result['stochastic_gpu_status']=status()
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
