import torch,json,pathlib
from ppo import physics
out={}
for mode in (False,True):
    s=torch.zeros(1,6,dtype=torch.float64,requires_grad=True)
    j=torch.autograd.functional.jacobian(lambda x:physics(x,torch.zeros(1,dtype=x.dtype),mode)[0],s).reshape(6,6)
    eig=torch.linalg.eigvals(j)
    out['physical' if mode else 'source']={'discrete_linearized_eigenvalues':[[v.real.item(),v.imag.item()] for v in eig],'spectral_radius':eig.abs().max().item(),'jacobian':j.tolist()}
p=pathlib.Path(__file__).resolve().parents[1]/'results/dynamics_diagnostic.json';p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
