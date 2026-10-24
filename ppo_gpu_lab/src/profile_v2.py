import torch
from torch.profiler import profile,ProfilerActivity
from ppo_v2 import TrainerV2
torch.set_num_threads(1)
t=TrainerV2();t.prepare()
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
 t.step();torch.cuda.synchronize()
print(p.key_averages().table(sort_by='self_cuda_time_total',row_limit=15))
