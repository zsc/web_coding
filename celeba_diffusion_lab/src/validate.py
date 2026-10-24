"""Numerical safeguards for the accelerated sampler and data protocol."""
import json
from pathlib import Path
import torch
from diffusers import DDIMScheduler
from model import new_model, Sampler

torch.set_num_threads(4); torch.manual_seed(49); torch.backends.cudnn.benchmark=True
m=new_model().cuda().to(memory_format=torch.channels_last).eval().requires_grad_(False)
with torch.no_grad():
    captured=Sampler(m,batch=2,steps=50,amp=False,graph=True)
    eager=Sampler(m,batch=2,steps=50,amp=False,graph=False)
    x=captured.run(seed=91); y=eager.run(seed=91)
    maximum=(x-y).abs().max().item(); assert maximum<3e-4,maximum
    # First transition is t=999 -> 979. Use the library's independently implemented scheduler.
    scheduler=DDIMScheduler(num_train_timesteps=1000,beta_start=.0001,beta_end=.02,
                            beta_schedule='linear',clip_sample=True,prediction_type='epsilon')
    scheduler.set_timesteps(50)
    eager.pos.zero_(); eager.x.normal_(); original=eager.x.clone()
    eps=m(original,torch.full((2,),999,device='cuda',dtype=torch.long),return_dict=False)[0]
    oracle=scheduler.step(eps,999,original,eta=0.,use_clipped_model_output=True).prev_sample
    eager._step(); oracle_error=(oracle-eager.x).abs().max().item()
    assert oracle_error<2e-5,oracle_error
    same=(captured.run(seed=91)-x).abs().max().item(); assert same==0,same
    different=(captured.run(seed=92)-x).abs().mean().item(); assert different>1e-3,different
    stochastic=Sampler(m,batch=2,steps=50,eta=1.,amp=True,graph=True)
    a=stochastic.run(91); b=stochastic.run(92); assert torch.isfinite(a).all() and torch.isfinite(b).all()
result={'graph_vs_eager_max_abs_fp32':maximum,'tolerance':3e-4,
        'library_ddim_first_step_max_abs':oracle_error,'library_tolerance':2e-5,
        'repeat_seed_max_abs':same,'different_seed_mean_abs':different,'stochastic_graph_finite':True}
Path('results/validation.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))
