"""One read-only-isolated GEMM counter probe; never changes monitoring/driver."""
import torch
x=torch.randn(256,256,device='cuda');y=torch.randn_like(x)
for _ in range(8):z=x@y
torch.cuda.synchronize();torch.cuda.profiler.start()
z=x@y
torch.cuda.synchronize();torch.cuda.profiler.stop()
assert torch.isfinite(z).all()
