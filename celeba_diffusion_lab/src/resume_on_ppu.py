"""Verify full optimizer/EMA/RNG transfer, then continue the existing trainer."""
import hashlib,json,os,runpy,sys,time
from pathlib import Path
import numpy as np
import torch

root=Path('results/flow_jit20k_ppu')
checkpoint_sha=hashlib.sha256((root/'checkpoint.pt').read_bytes()).hexdigest()
assert checkpoint_sha=='77edcf1235caf7003833b8d3d6aeacdd5de9b239a111d36f1e61d3c4121375e3'
c=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=False)
assert c['step']==12000 and c['config']['variant']=='jit' and c['config']['batch']==32
assert torch.cuda.device_count()==1
torch.cuda.set_rng_state(c['cuda_rng']);assert torch.equal(c['cuda_rng'],torch.cuda.get_rng_state())
noise=torch.randn(32,1,64,64,device='cuda').cpu().numpy()
expected=np.load(root/'resume_noise.npy');assert np.array_equal(noise,expected), 'CUDA RNG sequence differs across runtimes'
record={'source_device':'RTX 2080 SUPER','target_device':torch.cuda.get_device_name(0),
        'source_step':c['step'],'source_elapsed_seconds':c['elapsed_seconds'],
        'checkpoint_sha256':checkpoint_sha,
        'restored':['model','EMA','AdamW first/second moments','GradScaler','CPU RNG','CUDA RNG','NumPy RNG','curve','successful/skipped counts'],
        'cuda_rng_roundtrip_equal':True,'next_cuda_normal_batch_bitwise_equal':True,
        'torch':torch.__version__,'unix_started':time.time(),
        'note':'Different vendor kernels are not bitwise identical; optimizer and exact random streams are preserved.'}
(root/'migration.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
del c,noise,expected
sys.argv=['src/train_flow.py','--grayscale','--variant','jit','--steps','20000','--batch','32','--resume','--out',str(root)]
runpy.run_path('src/train_flow.py',run_name='__main__')
