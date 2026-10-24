"""Matched grayscale A/B/C diagnostics; no per-image output correction."""
import gc,json,time
from pathlib import Path
import numpy as np
import torch
from model import new_model,Sampler,save_grid,gpu_status
from evaluate_gray import summaries,compare,luminance,dc_probe

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    out=Path('results/affine_evaluation');out.mkdir(parents=True,exist_ok=True)
    old=json.loads(Path('results/gray_evaluation/distribution.json').read_text())
    result={k:old[k] for k in ('samples_per_method','real_count','real','grayscale_rule','units','reference','calibration','note')}
    result.update(rows=[r for r in old['rows'] if r['name'].startswith('gray5k')],models={},before=gpu_status())
    result['models']['baseline']={'dc_probe':old['gray_dc_probe'],'amp_graph_vs_fp32_eager_rmse':old['gray_amp_graph_vs_fp32_eager_rmse'],
                                 'training':json.loads(Path('results/gray5k/training.json').read_text())}
    valid=np.load('data/valid_gray.npy',mmap_mode='r');idx=np.linspace(0,len(valid)-1,2048,dtype=np.int64)
    real=torch.from_numpy(np.array(valid[idx])).permute(0,3,1,2).float()/127.5-1
    assert summaries(luminance(real))==result['real']
    save_grid(real[:64],out/'real_gray_64.png')
    for row in result['rows']:
        import shutil
        shutil.copyfile(Path('results/gray_evaluation')/(row['name']+'_64.png'),out/(row['name']+'_64.png'))
    for variant in ('time_affine','global_affine'):
        directory=Path('results')/(variant+'5k');checkpoint=directory/'checkpoint.pt'
        c=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert c['step']==5000 and c['config']['variant']==variant and c['config']['grayscale'] and c['config']['dc_weight']==0
        m=new_model(1,variant).cuda().eval().requires_grad_(False).to(memory_format=torch.channels_last)
        m.load_state_dict(c['ema'])
        md={'dc_probe':dc_probe(m,real),'training':json.loads((directory/'training.json').read_text())}
        for eta in (0.,1.):
            name=variant+'5k_eta'+str(int(eta));start=time.perf_counter()
            sampler=Sampler(m,batch=16,steps=100,eta=eta)
            images=torch.cat([sampler.run(seed=881+j) for j in range(32)])
            assert torch.isfinite(images).all();y=luminance(images)
            save_grid(y[:64]/127.5-1,out/(name+'_64.png'));torch.save(images,out/(name+'.pt'))
            s=summaries(y)
            row={'name':name,'training_channels':1,'training_steps':5000,'checkpoint':str(checkpoint),
                 'sampling_steps':100,'eta':eta,'seeds':list(range(881,913)),
                 'stats':s,'distance':compare(s,result['real']),'seconds':time.perf_counter()-start}
            result['rows'].append(row)
            print(json.dumps({'name':name,'mean':s['mean_brightness'],'std':s['image_brightness_std'],
                              'contrast':s['mean_contrast'],'distance':row['distance']}),flush=True)
            del sampler;gc.collect();torch.cuda.empty_cache()
        sam=Sampler(m,batch=4,steps=100,graph=True,amp=True);a=sam.run(3341)
        del sam;gc.collect();torch.cuda.empty_cache()
        sam=Sampler(m,batch=4,steps=100,graph=False,amp=False);b=sam.run(3341)
        md['amp_graph_vs_fp32_eager_rmse']=(a-b).square().mean().sqrt().item()
        result['models'][variant]=md
        del sam,m,c;gc.collect();torch.cuda.empty_cache()
        (out/'distribution.json').write_text(json.dumps(result,indent=2))
    # Interleave warm runs so clock/load drift is less confounded with architecture.
    samplers={};timings={v:[] for v in result['models']}
    for variant,directory in [('baseline','gray5k'),('time_affine','time_affine5k'),('global_affine','global_affine5k')]:
        c=torch.load(Path('results')/directory/'checkpoint.pt',map_location='cpu',weights_only=False)
        m=new_model(1,variant).cuda().eval().requires_grad_(False).to(memory_format=torch.channels_last);m.load_state_dict(c['ema'])
        samplers[variant]=Sampler(m,batch=5,steps=100,eta=0);samplers[variant].run(881)
        del c
    rng=np.random.default_rng(551)
    for rep in range(7):
        for variant in rng.permutation(list(samplers)):
            torch.cuda.synchronize();start=time.perf_counter();samplers[variant].run(881+rep);torch.cuda.synchronize()
            timings[variant].append(1000*(time.perf_counter()-start))
    result['timings']={v:{'ms':ts,'median_ms':float(np.median(ts))} for v,ts in timings.items()}
    result['timing_protocol']='batch 5, DDIM 100 eta=0, AMP/CUDA Graph, 7 interleaved warm runs; includes result CPU copy, excludes graph setup'
    result['after']=gpu_status()
    # All models use the same GPU stochastic stream and NumPy example stream.
    checks={};states=[]
    for name in ('gray5k','time_affine5k','global_affine5k'):
        c=torch.load(Path('results')/name/'checkpoint.pt',map_location='cpu',weights_only=False)
        states.append((c['cuda_rng'],c['numpy_rng']));del c
    checks['cuda_rng_equal']=all(torch.equal(states[0][0],s[0]) for s in states[1:])
    checks['numpy_rng_equal']=all(states[0][1]==s[1] for s in states[1:]);result['training_stream_checks']=checks
    (out/'distribution.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'event':'complete','timings':result['timings'],'training_stream_checks':checks}),flush=True)

if __name__=='__main__':main()
