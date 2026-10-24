"""RGB-only model comparison: fixed samples, v-loss, KID and sampling costs."""
import gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from model import save_grid,gpu_status
from model_flow import new_flow_model,FlowSampler

def write(p,d):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2));temp.replace(p)
def quantize(x):return ((x.float().clamp(-1,1)+1)*127.5).round()/255

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    out=Path('results/rgb_flow_evaluation');out.mkdir(parents=True,exist_ok=True)
    result={'channels':3,'real_count':2048,'samples_per_method':512,'rows':[],'models':{},
            'before':gpu_status(),'torch':torch.__version__,
            'protocol':'Original RGB, no grayscale conversion or color correction; Heun50 intervals/final Euler,99NFE; seeds881..912; first64 displayed,all512 evaluated.'}
    valid=np.load('data/valid.npy',mmap_mode='r');idx=np.linspace(0,len(valid)-1,2048,dtype=np.int64)
    real=torch.from_numpy(np.array(valid[idx])).permute(0,3,1,2).float()/127.5-1
    save_grid(real[:64],out/'real_rgb_64.png')
    for kind,key,label,steps in [('unet','unet','flow_unet5k',5000),('jit','jit5k','flow_jit5k',5000),('jit','jit','flow_jit20k',20000)]:
        directory=Path('results')/('rgb_flow_unet5k' if kind=='unet' else 'rgb_flow_jit20k')
        weights=directory/f'ema_{steps}.pt';c=torch.load(weights,map_location='cpu',weights_only=False)
        assert c['step']==steps and c['config']['variant']==kind and not c['config']['grayscale']
        model=new_flow_model(kind,3).cuda().eval().requires_grad_(False).to(memory_format=torch.channels_last);model.load_state_dict(c['ema'])
        sampler=FlowSampler(model,batch=16,steps=50);start=time.perf_counter()
        images=torch.cat([sampler.run(881+j) for j in range(32)])
        assert images.shape==(512,3,64,64) and torch.isfinite(images).all()
        torch.save(images,out/(label+'.pt'));save_grid(images[:64],out/(label+'_64.png'))
        row={'name':label,'training_steps':steps,'seeds':list(range(881,913)),
             'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
             'outside_unit_fraction':(images.abs()>1).float().mean().item(),'seconds':time.perf_counter()-start}
        result['rows'].append(row);print(json.dumps(row),flush=True)
        del sampler;gc.collect();torch.cuda.empty_cache()
        md={'training':json.loads((directory/f'training_{steps}.json').read_text())}
        sampler=FlowSampler(model,batch=4,steps=50,graph=True,amp=True);a=sampler.run(3341)
        del sampler;gc.collect();torch.cuda.empty_cache()
        sampler=FlowSampler(model,batch=4,steps=50,graph=False,amp=False);b=sampler.run(3341)
        md['amp_graph_vs_fp32_eager_rmse']=(a-b).square().mean().sqrt().item()
        save_grid(b,out/(label+'_fp32_4.png'),columns=4,scale=3);result['models'][key]=md
        del sampler,model,c;gc.collect();torch.cuda.empty_cache();write(out/'evaluation.json',result)
    samplers={};values={}
    for name,kind,steps in [('flow_unet','unet',5000),('flow_jit','jit',20000)]:
        directory='rgb_flow_unet5k' if kind=='unet' else 'rgb_flow_jit20k'
        c=torch.load(f'results/{directory}/ema_{steps}.pt',map_location='cpu',weights_only=False)
        model=new_flow_model(kind,3).cuda().eval().requires_grad_(False).to(memory_format=torch.channels_last);model.load_state_dict(c['ema'])
        sampler=FlowSampler(model,batch=5,steps=50);sampler.run(881);samplers[name]=sampler;values[name]=[]
    rng=np.random.default_rng(551)
    for rep in range(7):
        for name in rng.permutation(list(samplers)):
            torch.cuda.synchronize();start=time.perf_counter();samplers[name].run(881+rep);torch.cuda.synchronize();values[name].append(1000*(time.perf_counter()-start))
    result['timings']={k:{'ms':v,'median_ms':float(np.median(v))} for k,v in values.items()}
    result['timing_protocol']='Same PPU;7 interleaved warm runs,batch5,99NFE;includes CPU output copy,excludes graph capture/PNG'
    result['after']=gpu_status();del samplers,sampler,model,c;gc.collect();torch.cuda.empty_cache()
    cs=[torch.load(f'results/{v}/rng_5000.pt',map_location='cpu',weights_only=False) for v in ('rgb_flow_unet5k','rgb_flow_jit20k')]
    result['training_stream_checks']={'cuda_rng_equal':torch.equal(cs[0]['cuda_rng'],cs[1]['cuda_rng']),'numpy_rng_equal':cs[0]['numpy_rng']==cs[1]['numpy_rng']}
    write(out/'evaluation.json',result)
    from pytorch_fid.inception import InceptionV3
    net=InceptionV3([3]).cuda().eval().requires_grad_(False)
    def features(images):
        x=quantize(images);assert x.shape[1]==3
        return np.concatenate([net(x[i:i+32].cuda())[0].flatten(1).cpu().numpy() for i in range(0,len(x),32)])
    rf=features(real);np.save(out/'real_features.npy',rf);rows=[]
    for row in result['rows']:
        name=row['name'];fake=features(torch.load(out/(name+'.pt'),map_location='cpu',weights_only=True));np.save(out/(name+'_features.npy'),fake)
        rng=np.random.default_rng(42);values=[];n=256;dim=rf.shape[1]
        for _ in range(50):
            a=rf[rng.choice(len(rf),n,replace=False)].astype(np.float64);b=fake[rng.choice(len(fake),n,replace=False)].astype(np.float64)
            aa=(a@a.T/dim+1)**3;bb=(b@b.T/dim+1)**3;ab=(a@b.T/dim+1)**3
            values.append(float((aa.sum()-aa.trace()+bb.sum()-bb.trace())/(n*(n-1))-2*ab.mean()))
        r={'name':name,'kid_mean':float(np.mean(values)),'subset_std':float(np.std(values,ddof=1)),'values':values};rows.append(r)
        print(json.dumps({'kid':name,'mean_x1000':r['kid_mean']*1000,'sd_x1000':r['subset_std']*1000}),flush=True)
        write(out/'kid.json',{'real_count':2048,'generated_per_method':512,'subsets':50,'subset_size':256,'features':'pytorch-fid0.3.0 Inception pool3;original RGB,quantized identically',
              'note':'Subset SD is not a confidence interval;one training seed;20k JiT uses4x the5k exposure. Do not mix with historical grayscale KID.','results':rows})
    print(json.dumps({'event':'complete','streams':result['training_stream_checks']}),flush=True)

if __name__=='__main__':main()
