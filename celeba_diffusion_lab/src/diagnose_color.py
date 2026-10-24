"""Controlled sampler ablations and color/DC error diagnostics, without recoloring outputs."""
import argparse, gc, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from model import new_model,Sampler,save_grid,alpha_bar

def color_stats(x):
    y=(x.detach().cpu().float().clamp(-1,1)+1)*127.5
    means=y.mean((2,3)); center=y[:,:,16:48,16:48].mean((2,3))
    return {'n':len(y),'rgb_mean':means.mean(0).tolist(),'per_image_rgb_std':means.std(0).tolist(),
            'red_minus_blue':(means[:,0]-means[:,2]).mean().item(),
            'blue_dominant_fraction':(means[:,2]>means[:,0]).float().mean().item(),
            'center_rgb_mean':center.mean(0).tolist(),'per_image_rgb_means':means.tolist(),
            'final_outside_unit_fraction':(x.abs()>1).float().mean().item()}

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='results/run1/ema_5000.pt')
    p.add_argument('--out',default='results/color_diagnosis');p.add_argument('--samples',type=int,default=16)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    c=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    m=new_model().cuda().eval().requires_grad_(False).to(memory_format=torch.channels_last);m.load_state_dict(c['ema'])
    valid=np.load('data/valid.npy',mmap_mode='r');indices=np.linspace(0,len(valid)-1,2048,dtype=np.int64)
    real=torch.from_numpy(np.array(valid[indices])).permute(0,3,1,2).float()/127.5-1
    # The RGB decode/normalization/display path must round-trip exactly.
    inv=((real+1)*127.5).round().byte().permute(0,2,3,1).numpy()
    roundtrip=int(np.abs(inv.astype(np.int16)-valid[indices].astype(np.int16)).max());assert roundtrip==0
    torch.manual_seed(881)
    initial=torch.empty(args.samples,3,64,64,device='cuda',memory_format=torch.channels_last).normal_()
    configs=[('clip_corrected',True,True,0.,True,True),('clip_keep_epsilon',True,False,0.,True,True),
             ('unclipped',False,False,0.,True,True),('stochastic',True,True,1.,True,True),
             ('fp32_eager',True,True,0.,False,False)]
    rows=[];outputs={}
    for name,clip,correct,eta,amp,graph in configs:
        sampler=Sampler(m,batch=4,steps=100,eta=eta,amp=amp,graph=graph,clip=clip,correct_eps=correct)
        ims=torch.cat([sampler.run(seed=881+j//4,initial=initial[j:j+4]) for j in range(0,args.samples,4)])
        assert torch.isfinite(ims).all();save_grid(ims,out/(name+'.png'),columns=8 if args.samples>=64 else 4,scale=2 if args.samples>=64 else 3)
        torch.save(ims,out/(name+'.pt'));outputs[name]=ims
        row={'name':name,'clip':clip,'correct_eps':correct,'eta':eta,'amp':amp,'graph':graph,**color_stats(ims)}
        rows.append(row);print(json.dumps(row),flush=True);del sampler;gc.collect();torch.cuda.empty_cache()
    # DC = spatial mean in each channel. A small DC error in epsilon can amplify at high noise.
    dc=[];ab=alpha_bar()
    for timestep in (100,500,900,999):
        errors=[];clean_errors=[];clipping=[];naive_errors=[];predicted=[];truth=[]
        generator=torch.Generator(device='cuda').manual_seed(2401)
        for i in range(0,128,8):
            x0=real[i:i+8].cuda().to(memory_format=torch.channels_last)
            noise=torch.randn(x0.shape,device='cuda',generator=generator)
            a=ab[timestep];xt=a.sqrt()*x0+(1-a).sqrt()*noise
            t=torch.full((len(x0),),timestep,device='cuda',dtype=torch.long)
            pred=m(xt,t,return_dict=False)[0]
            errors.append((pred-noise).mean((2,3)).cpu())
            # Baseline that simply copies the observable noisy image mean (x0-mean assumed zero).
            naive_errors.append((xt.mean((2,3))/(1-a).sqrt()-noise.mean((2,3))).cpu())
            predicted.append(pred.mean((2,3)).cpu()); truth.append(noise.mean((2,3)).cpu())
            estimated=(xt-(1-a).sqrt()*pred)/a.sqrt()
            clean_errors.append((estimated-x0).mean((2,3)).cpu())
            clipping.append((estimated.abs()>1).float().mean().item())
        err=torch.cat(errors);errx=torch.cat(clean_errors)
        dc.append({'t':timestep,'epsilon_dc_error_mean_rgb':err.mean(0).tolist(),
                   'epsilon_dc_error_rmse':err.square().mean().sqrt().item(),
                   'x0_dc_error_rmse':errx.square().mean().sqrt().item(),
                   'copy_noisy_mean_epsilon_dc_rmse':torch.cat(naive_errors).square().mean().sqrt().item(),
                   'predicted_epsilon_dc_std_rgb':torch.cat(predicted).std(0).tolist(),
                   'true_epsilon_dc_std_rgb':torch.cat(truth).std(0).tolist(),
                   'x0_outside_unit_fraction':float(np.mean(clipping)),
                   'epsilon_to_x0_amplification':((1-ab[timestep])/ab[timestep]).sqrt().item()})
    result={'step':c['step'],'checkpoint':args.checkpoint,'seed':881,'samples':args.samples,'sample_batch':4,
            'real_reference':'2048 evenly spaced official validation images','roundtrip_max_uint8_error':roundtrip,
            'real':color_stats(real),'samplers':rows,'dc_diagnostics':dc,
            'amp_graph_vs_fp32_eager_rmse':(outputs['clip_corrected']-outputs['fp32_eager']).square().mean().sqrt().item(),
            'note':'Color measurements in displayed 0..255 RGB. Unclipped trajectories are only clamped for display/statistics; saved tensors preserve raw output. No white balance or distribution matching postprocessing.'}
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
