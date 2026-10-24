"""Train-gray vs train-RGB luminance distributions; never normalize generated samples."""
import argparse, gc, json, time
from pathlib import Path
import numpy as np
import torch
from model import new_model, Sampler, save_grid, alpha_bar


def luminance(x):
    """Same display quantization and grayscale rule as prepare_gray.py."""
    y=((x.detach().cpu().float().clamp(-1,1)+1)*127.5).round()
    if y.shape[1]==3:
        y=(y*torch.tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True).round()
    return y


def summaries(y):
    flat=y[:,0].numpy().astype(np.float64).reshape(len(y),-1)
    means=flat.mean(1); contrast=flat.std(1)
    hist=np.bincount(flat.astype(np.uint8).reshape(-1),minlength=256).astype(float)
    return {'n':len(y),'mean_brightness':float(means.mean()),'image_brightness_std':float(means.std(ddof=1)),
            'mean_contrast':float(contrast.mean()),'brightness_quantiles':np.percentile(means,[5,25,50,75,95]).tolist(),
            'contrast_quantiles':np.percentile(contrast,[5,25,50,75,95]).tolist(),
            'brightness':means.tolist(),'contrast':contrast.tolist(),'pixel_histogram':(hist/hist.sum()).tolist()}


def wasserstein(a,b):
    a=np.sort(np.asarray(a));b=np.sort(np.asarray(b));allv=np.sort(np.concatenate((a,b)))
    return float(np.sum(np.abs(np.searchsorted(a,allv[:-1],side='right')/len(a)-
                               np.searchsorted(b,allv[:-1],side='right')/len(b))*np.diff(allv)))


def compare(s,ref):
    lo,hi=ref['brightness_quantiles'][0],ref['brightness_quantiles'][-1]
    return {'brightness_wasserstein':wasserstein(s['brightness'],ref['brightness']),
            'contrast_wasserstein':wasserstein(s['contrast'],ref['contrast']),
            'outside_real_brightness_p5_p95':float(np.mean((np.array(s['brightness'])<lo)|(np.array(s['brightness'])>hi))),
            'pixel_wasserstein':float(np.abs(np.cumsum(s['pixel_histogram'])-np.cumsum(ref['pixel_histogram'])).sum())}


@torch.no_grad()
def dc_probe(m,real):
    ab=alpha_bar();rows=[]
    for t in (100,500,900,999):
        errors=[];naive=[];clean=[];preds=[];truth=[];outside=[];gains=[]
        rng=torch.Generator(device='cuda').manual_seed(2401)
        for j in range(0,128,8):
            x0=real[j:j+8].cuda().to(memory_format=torch.channels_last)
            noise=torch.randn(x0.shape,device='cuda',generator=rng);a=ab[t]
            xt=a.sqrt()*x0+(1-a).sqrt()*noise
            eps=m(xt,torch.full((len(x0),),t,device='cuda',dtype=torch.long),return_dict=False)[0]
            errors.append((eps-noise).mean((2,3)).cpu())
            naive.append((xt.mean((2,3))/(1-a).sqrt()-noise.mean((2,3))).cpu())
            estimated=(xt-(1-a).sqrt()*eps)/a.sqrt()
            clean.append((estimated-x0).mean((2,3)).cpu());outside.append((estimated.abs()>1).float().mean().item())
            preds.append(eps.mean((2,3)).cpu());truth.append(noise.mean((2,3)).cpu())
            if t>=900:
                ts=torch.full((len(x0),),t,device='cuda',dtype=torch.long)
                plus=m(xt+.01,ts,return_dict=False)[0]
                minus=m(xt-.01,ts,return_dict=False)[0]
                gains.append(((plus-minus)/.02).mean((1,2,3)).cpu())
        rows.append({'t':t,'epsilon_dc_rmse':torch.cat(errors).square().mean().sqrt().item(),
                     'copy_noisy_mean_epsilon_dc_rmse':torch.cat(naive).square().mean().sqrt().item(),
                     'x0_dc_rmse':torch.cat(clean).square().mean().sqrt().item(),
                     'predicted_dc_std':torch.cat(preds).std(0).tolist(),'true_dc_std':torch.cat(truth).std(0).tolist(),
                     'x0_outside_unit_fraction':float(np.mean(outside)),
                     'amplification':((1-a)/a).sqrt().item(),
                     'constant_input_shift_response_gain':torch.cat(gains).mean().item() if gains else None})
    return rows


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=int,default=512)
    p.add_argument('--out',default='results/gray_evaluation');args=p.parse_args()
    assert args.samples>=64 and args.samples%16==0
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    valid=np.load('data/valid_gray.npy',mmap_mode='r')
    idx=np.linspace(0,len(valid)-1,2048,dtype=np.int64)
    real=torch.from_numpy(np.array(valid[idx])).permute(0,3,1,2).float()
    ref=summaries(real);save_grid(real[:64]/127.5-1,out/'real_gray_64.png')
    # The reference-vs-reference calibration uses a separate validation pool.
    pool=np.setdiff1d(np.arange(len(valid)),idx);rng=np.random.default_rng(451)
    pool_flat=np.asarray(valid[pool]).reshape(len(pool),-1)
    pool_mean=pool_flat.mean(1);pool_contrast=pool_flat.std(1)
    null_b=[];null_c=[]
    for _ in range(200):
        pick=rng.choice(len(pool),args.samples,replace=False)
        null_b.append(wasserstein(pool_mean[pick],ref['brightness']))
        null_c.append(wasserstein(pool_contrast[pick],ref['contrast']))
    del pool_flat
    result={'samples_per_method':args.samples,'real_count':2048,'real':ref,'rows':[],
            'grayscale_rule':'round(0.299 R + 0.587 G + 0.114 B) after RGB uint8 resize; no recentering or rescaling',
            'units':'0..255 display values; image brightness = spatial mean, contrast = within-image population SD',
            'reference':'2048 evenly spaced official validation images; no test data',
            'calibration':{'repeats':200,'samples':args.samples,'reference_pool_disjoint':True,
                           'brightness_wasserstein_p5_p50_p95':np.percentile(null_b,[5,50,95]).tolist(),
                           'contrast_wasserstein_p5_p50_p95':np.percentile(null_c,[5,50,95]).tolist()},
            'note':'Descriptive marginal checks, not proof of full-distribution equality or a Gaussianity test. One training seed; reference calibration is not a model uncertainty interval.'}
    for label,checkpoint,channels in [('gray5k','results/gray5k/checkpoint.pt',1),('rgb5k','results/run1/ema_5000.pt',3)]:
        c=torch.load(checkpoint,map_location='cpu',weights_only=False);assert c['step']==5000
        m=new_model(channels=channels).cuda().eval().requires_grad_(False).to(memory_format=torch.channels_last)
        m.load_state_dict(c['ema'])
        if channels==1:
            assert c['config']['grayscale'];result['gray_parameters']=sum(p.numel() for p in m.parameters())
            norm=real/127.5-1
            result['roundtrip_uint8_max_error']=int((luminance(norm)-real).abs().max());assert result['roundtrip_uint8_max_error']==0
            result['gray_dc_probe']=dc_probe(m,norm)
        for eta in (0.,1.):
            name=label+'_eta'+str(int(eta));start=time.perf_counter();sampler=Sampler(m,batch=16,steps=100,eta=eta)
            images=torch.cat([sampler.run(seed=881+j) for j in range(args.samples//16)])
            assert torch.isfinite(images).all();y=luminance(images)
            save_grid(y[:64]/127.5-1,out/(name+'_64.png'));torch.save(images,out/(name+'.pt'))
            s=summaries(y)
            row={'name':name,'training_channels':channels,'training_steps':c['step'],'checkpoint':checkpoint,
                 'sampling_steps':100,'eta':eta,'seeds':list(range(881,881+args.samples//16)),
                 'stats':s,'distance':compare(s,ref),'seconds':time.perf_counter()-start}
            result['rows'].append(row)
            print(json.dumps({'name':name,'seconds':row['seconds'],'mean':s['mean_brightness'],'std':s['image_brightness_std'],
                              'contrast':s['mean_contrast'],'distance':row['distance']}),flush=True)
            del sampler;gc.collect();torch.cuda.empty_cache()
            (out/'distribution.json').write_text(json.dumps(result,indent=2))
        if channels==1:
            sam=Sampler(m,batch=4,steps=100,graph=True,amp=True);a=sam.run(3341)
            del sam;gc.collect();torch.cuda.empty_cache()
            sam=Sampler(m,batch=4,steps=100,graph=False,amp=False);b=sam.run(3341)
            result['gray_amp_graph_vs_fp32_eager_rmse']=(a-b).square().mean().sqrt().item()
            save_grid(b,out/'gray_fp32_4.png',columns=4,scale=3)
            del sam;gc.collect();torch.cuda.empty_cache()
        del m,c;gc.collect();torch.cuda.empty_cache()
    (out/'distribution.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'event':'complete','out':str(out)}),flush=True)

if __name__=='__main__':main()
