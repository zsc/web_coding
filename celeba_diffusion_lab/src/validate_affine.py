"""GPU checks of conditioning, trainability, serialization and CUDA capture."""
import copy,gc,io,json
from pathlib import Path
import torch
from model import new_model,Sampler

torch.set_num_threads(4);torch.manual_seed(20261024)
b=new_model(1,'time_affine').cuda().to(memory_format=torch.channels_last)
torch.manual_seed(20261024)
c=new_model(1,'global_affine').cuda().to(memory_format=torch.channels_last)
assert all(torch.equal(v,c.state_dict()[k]) for k,v in b.state_dict().items())
x=torch.randn(4,1,64,64,device='cuda').to(memory_format=torch.channels_last)
t=torch.full((4,),999,device='cuda',dtype=torch.long)
with torch.no_grad():
    initial_error=(b(x,t,return_dict=False)[0]-c(x,t,return_dict=False)[0]).abs().max().item()
assert initial_error==0
opt=torch.optim.AdamW(c.parameters(),lr=2e-4)
losses=[]
for _ in range(3):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):
        loss=(c(x,t,return_dict=False)[0].float()-x).square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in c.parameters())
    torch.nn.utils.clip_grad_norm_(c.parameters(),1);opt.step();losses.append(loss.item())
assert c.global_encoder[0].weight.grad.abs().sum()>0
with torch.no_grad():
    sensitivity=(c.global_condition(x+.01)-c.global_condition(x)).abs().max().item()
assert sensitivity>0
buf=io.BytesIO();torch.save(c.state_dict(),buf);buf.seek(0)
restored=new_model(1,'global_affine').cuda().to(memory_format=torch.channels_last)
restored.load_state_dict(torch.load(buf,weights_only=True))
with torch.no_grad():
    save_error=(c(x,t,return_dict=False)[0]-restored(x,t,return_dict=False)[0]).abs().max().item()
assert save_error==0
sam=Sampler(c,batch=4,steps=20,graph=False,amp=False);a=sam.run(3341)
del sam;gc.collect();torch.cuda.empty_cache()
sam=Sampler(c,batch=4,steps=20,graph=True,amp=False);z=sam.run(3341)
err=(a-z).abs().max().item();assert err<1e-5
result={'shared_initial_weights_equal':True,'initial_output_max_error':initial_error,
        'training_losses':losses,'condition_input_sensitivity':sensitivity,
        'global_encoder_gradient_nonzero':True,'save_load_max_error':save_error,
        'fp32_graph_vs_eager_max_error':err,
        'parameters':{v:sum(p.numel() for p in new_model(1,v).parameters()) for v in ('baseline','time_affine','global_affine')}}
Path('results/affine_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
