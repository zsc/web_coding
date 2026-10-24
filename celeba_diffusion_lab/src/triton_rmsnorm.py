"""FP32-accumulating, non-affine RMSNorm for JiT residual streams only.

Q/K norms with learned weights keep the original implementation. No GEMM or
attention replacement, dtype change, approximate reciprocal, or fast math.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(X, Y, R, M: tl.constexpr, N: tl.constexpr, EPS: tl.constexpr,
             C: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0)*ROWS+tl.arange(0, ROWS)
    cols = tl.arange(0, C)
    x = tl.load(X+rows[:, None]*N+cols[None, :], (rows[:, None]<M)&(cols[None, :]<N), other=0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x*x, 1)/N+EPS)
    tl.store(R+rows, r, rows<M)
    tl.store(Y+rows[:, None]*N+cols[None, :], x*r[:, None], (rows[:, None]<M)&(cols[None, :]<N))


@triton.jit
def _backward(X, G, R, DX, M: tl.constexpr, N: tl.constexpr,
              C: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0)*ROWS+tl.arange(0, ROWS)
    cols = tl.arange(0, C)
    mask = (rows[:, None]<M)&(cols[None, :]<N)
    x = tl.load(X+rows[:, None]*N+cols[None, :], mask, other=0).to(tl.float32)
    g = tl.load(G+rows[:, None]*N+cols[None, :], mask, other=0).to(tl.float32)
    r = tl.load(R+rows, rows<M, other=0)
    dot = tl.sum(g*x, 1)/N
    dx = g*r[:, None]-x*(dot*r*r*r)[:, None]
    tl.store(DX+rows[:, None]*N+cols[None, :], dx, mask)


class _RMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        assert x.is_contiguous(), 'Only tested contiguous residual stream inputs are supported'
        n=x.shape[-1]; m=x.numel()//n
        y=torch.empty_like(x); r=torch.empty(m,device=x.device,dtype=torch.float32)
        _forward[(triton.cdiv(m,4),)](x,y,r,m,n,eps,triton.next_power_of_2(n),4,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(x,r)
        return y

    @staticmethod
    def backward(ctx, grad):
        x,r=ctx.saved_tensors; g=grad.contiguous(); dx=torch.empty_like(x)
        n=x.shape[-1]; m=x.numel()//n
        _backward[(triton.cdiv(m,4),)](x,g,r,dx,m,n,triton.next_power_of_2(n),4,num_warps=4,enable_fp_fusion=False)
        return dx,None


class FastRMSNorm(torch.nn.Module):
    def __init__(self, eps):
        super().__init__(); self.eps=eps

    def forward(self, x):
        return _RMS.apply(x,self.eps)


def install(model):
    from vendor_jit.models.jit_stft import RMSNorm
    count=0
    for name, child in list(model.named_children()):
        if isinstance(child,RMSNorm) and child.weight is None:
            setattr(model,name,FastRMSNorm(child.eps)); count+=1
        else: count+=install(child)
    return count
