"""Direct-x flow comparisons: distribution, learned denoising, KID and costs."""
import gc,json,time
from pathlib import Path
import numpy as np
import torch
from model import save_grid,gpu_status
from model_flow import new_flow_model,FlowSampler
from evaluate_gray import summaries,compare,luminance

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    out=Path('results/flow_evaluation');out.mkdir(parents=True,exist_ok=True)
    af=json.loads(Path('results/affine_evaluation/distribution.json').read_text())
    result={k:af[k] for k in ('real','real_count','samples_per_method','calibration','units','reference','note')}
    result.update(rows=[],models={},before=gpu_status())
    valid=np.load('data/valid_gray.npy',mmap_mode='r');idx=np.linspace(0,len(valid)-1,2048,dtype=np.int64)
    real=torch.from_numpy(np.array(valid[idx])).permute(0,3,1,2).float()/127.5-1
    save_grid(real[:64],out/'real_gray_64.png')
    for kind,key,label,steps in [('unet','unet','flow_unet5k',5000),('jit','jit5k','flow_jit5k',5000),('jit','jit','flow_jit20k',20000)]:
        directory=Path('results')/('flow_unet5k' if kind=='unet' else 'flow_jit20k')
        c=torch.load(directory/f'ema_{steps}.pt',map_location='cpu',weights_only=False)
        assert c['step']==steps and c['config']['variant']==kind
        m=new_flow_model(kind).cuda().eval().requires_grad_(False).to(memory_format=torch.channels_last);m.load_state_dict(c['ema'])
        sampler=FlowSampler(m,batch=16,steps=50);start=time.perf_counter()
        images=torch.cat([sampler.run(881+j) for j in range(32)]);assert torch.isfinite(images).all()
        torch.save(images,out/(label+'.pt'));y=luminance(images);save_grid(y[:64]/127.5-1,out/(label+'_64.png'))
        s=summaries(y);row={'name':label,'training_steps':steps,'stats':s,'distance':compare(s,result['real']),
                           'seeds':list(range(881,913)),'sampling':'Heun 50 intervals, final Euler; 99 NFE',
                           'outside_unit_fraction':(images.abs()>1).float().mean().item(),'seconds':time.perf_counter()-start}
        result['rows'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
        del sampler;gc.collect();torch.cuda.empty_cache()
        md={'training':json.loads((directory/f'training_{steps}.json').read_text()),'probe':[]}
        sam=FlowSampler(m,batch=4,steps=50,graph=True,amp=True);a=sam.run(3341)
        del sam;gc.collect();torch.cuda.empty_cache()
        sam=FlowSampler(m,batch=4,steps=50,graph=False,amp=False);b=sam.run(3341)
        md['amp_graph_vs_fp32_eager_rmse']=(a-b).square().mean().sqrt().item()
        save_grid(b,out/(label+'_fp32_4.png'),columns=4,scale=3);result['models'][key]=md
        del sam,m,c;gc.collect();torch.cuda.empty_cache()
        (out/'distribution.json').write_text(json.dumps(result,indent=2))
    samplers={};values={}
    for name in ('flow_unet','flow_jit'):
        kind=name.split('_')[1]
        path='results/flow_unet5k/checkpoint.pt' if kind=='unet' else 'results/flow_jit20k/checkpoint.pt'
        c=torch.load(path,map_location='cpu',weights_only=False)
        m=new_flow_model(kind).cuda().eval().requires_grad_(False).to(memory_format=torch.channels_last)
        m.load_state_dict(c['ema']);sam=FlowSampler(m,batch=5,steps=50)
        sam.run(881);samplers[name]=sam;values[name]=[];del c
    rng=np.random.default_rng(551)
    for rep in range(7):
        for name in rng.permutation(list(samplers)):
            torch.cuda.synchronize();start=time.perf_counter();samplers[name].run(881+rep);torch.cuda.synchronize()
            values[name].append(1000*(time.perf_counter()-start))
    result['timings']={k:{'ms':v,'median_ms':float(np.median(v))} for k,v in values.items()}
    result['timing_protocol']='7 interleaved warm runs, batch5; includes CPU copy, excludes graph capture/PNG; flow=99 NFE'
    result['after']=gpu_status();del samplers,sam,m;gc.collect();torch.cuda.empty_cache()
    cs=[torch.load(f'results/{v}/rng_5000.pt',map_location='cpu',weights_only=False) for v in ('flow_unet5k','flow_jit20k')]
    result['training_stream_checks']={'cuda_rng_equal':torch.equal(cs[0]['cuda_rng'],cs[1]['cuda_rng']),
                                      'numpy_rng_equal':cs[0]['numpy_rng']==cs[1]['numpy_rng']}
    del cs;gc.collect()
    (out/'distribution.json').write_text(json.dumps(result,indent=2))
    run_kid(out,real,result)
    print(json.dumps({'event':'complete','timings':result['timings'],'streams':result['training_stream_checks']}),flush=True)

@torch.no_grad()
def run_kid(out,real,flow):
    from pytorch_fid.inception import InceptionV3
    net=InceptionV3([3]).cuda().eval().requires_grad_(False)
    def features(images):
        # Identical gray->3-channel repetition for real and generated, uint8 display codec.
        y=luminance(images)/255
        return np.concatenate([net(y[i:i+32].repeat(1,3,1,1).cuda())[0].flatten(1).cpu().numpy() for i in range(0,len(y),32)])
    rf=features(real);np.save(out/'real_features.npy',rf)
    rows=[]
    paths = [(r['name'],out/(r['name']+'.pt')) for r in flow['rows']]
    for name,path in paths:
        fake=features(torch.load(path,map_location='cpu',weights_only=True));np.save(out/(name+'_features.npy'),fake)
        rng=np.random.default_rng(42);values=[];n=256;d=rf.shape[1]
        for _ in range(50):
            a=rf[rng.choice(len(rf),n,replace=False)].astype(np.float64);b=fake[rng.choice(len(fake),n,replace=False)].astype(np.float64)
            aa=(a@a.T/d+1)**3;bb=(b@b.T/d+1)**3;ab=(a@b.T/d+1)**3
            values.append(float((aa.sum()-aa.trace()+bb.sum()-bb.trace())/(n*(n-1))-2*ab.mean()))
        row={'name':name,'kid_mean':float(np.mean(values)),'subset_std':float(np.std(values,ddof=1)),'values':values}
        rows.append(row);print(json.dumps({'kid':name,'mean_x1000':row['kid_mean']*1000,'sd_x1000':row['subset_std']*1000}),flush=True)
        (out/'kid.json').write_text(json.dumps({'real_count':2048,'generated_per_method':512,'subsets':50,'subset_size':256,
              'features':'pytorch-fid Inception pool3; quantized gray repeated to RGB, same preprocessing for all methods',
              'note':'Subset SD is not a confidence interval; one training seed; no comparison to older RGB KID or published FID.',
              'results':rows},indent=2))

if __name__=='__main__':main()
