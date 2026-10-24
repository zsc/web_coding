"""Design a discrete LQR for the SAME physical model and dt as GPU PPO.
No fitting, learning, or data-driven tuning. Force unit is newtons.
"""
import json,pathlib
import numpy as np
import torch
from scipy.linalg import solve_discrete_are
from ppo import physics

torch.set_default_dtype(torch.float64)
s=torch.zeros(6,requires_grad=True)
u=torch.zeros(1,requires_grad=True)
fn=lambda s,u:physics(s[None,:],u/10,True)[0][0]
A,B=torch.autograd.functional.jacobian(fn,(s,u))
A=A.detach().numpy();B=B.detach().numpy()
Q=np.diag([4.,2.,80.,4.,120.,4.]);R=np.array([[.15]])
P=solve_discrete_are(A,B,Q,R)
K=np.linalg.solve(R+B.T@P@B,B.T@P@A)
out={'method':'discrete LQR, exact autograd linearization at upright; u = clip(-K s, -10, 10) N',
     'dt':.02,'Q':Q.tolist(),'R':R.tolist(),'K':K[0].tolist(),
     'A':A.tolist(),'B':B.tolist(),'closed_loop_eigenvalues':[[z.real,z.imag] for z in np.linalg.eigvals(A-B@K)],
     'spectral_radius':float(max(abs(np.linalg.eigvals(A-B@K))))}
root=pathlib.Path(__file__).resolve().parents[1]
(root/'results/lqr_design.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
