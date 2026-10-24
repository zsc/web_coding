"""Gate 16-rank gradient averaging and graph/Triton optimizer trajectories."""
import argparse
import copy
import json
from pathlib import Path
import torch
import torch.distributed as dist
from distributed_acceleration import init, TrainingState, AverageGradients
from model_flow import flow_loss
from check_training_acceleration import diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=('unet', 'scaled'), required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--global-batch', type=int, default=512)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--seed', type=int, default=20261025)
    a = p.parse_args()
    rank, world = init()
    assert world == 16 and a.global_batch == 512
    ref = TrainingState(a, acceleration=False)
    # The semantic FP32 oracle must use the same deterministic full-precision
    # kernels across devices, rather than each device's autotuner selection.
    backend = {'benchmark': torch.backends.cudnn.benchmark,
               'deterministic': torch.backends.cudnn.deterministic,
               'cudnn_tf32': torch.backends.cudnn.allow_tf32,
               'matmul_tf32': torch.backends.cuda.matmul.allow_tf32}
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    batch = ref.inputs()
    gathered = []
    for item in batch:
        parts = [torch.empty_like(item) for _ in range(world)]
        dist.all_gather(parts, item)
        gathered.append(parts)
    ref.opt.zero_grad(set_to_none=True)
    flow_loss(ref.model(batch[2], batch[1]), batch[0], batch[1]).backward()
    # An independent FP64 reduction of the exact local gradients distinguishes
    # reducer errors from different per-device convolution algorithms.
    reduced64 = torch.cat([p.grad.detach().contiguous().flatten() for p in ref.params]).double()
    dist.all_reduce(reduced64, op=dist.ReduceOp.SUM)
    reduced64.div_(world)
    AverageGradients(ref.params)()
    actual64 = torch.cat([p.grad.detach().contiguous().flatten() for p in ref.params]).double()
    reducer_relative = float((actual64-reduced64).norm()/reduced64.norm())
    assert reducer_relative < 1e-6, reducer_relative
    del actual64, reduced64
    average_check = None
    if rank == 0:
        full = copy.deepcopy(ref.model)
        full.zero_grad(set_to_none=True)
        # Accumulating all 16 microbatches is the full-batch mean gradient.
        for j in range(world):
            x, t, z = [parts[j] for parts in gathered]
            (flow_loss(full(z, t), x, t)/world).backward()
        average_check = diff([p.grad for p in ref.params], [p.grad for p in full.parameters()])
        # FP32 vendor convolution choices can differ across devices. Retain a
        # separate absolute bound plus the tighter FP64 reducer check above.
        print(json.dumps({'event':'fp32_oracle','average_check':average_check,'reducer_vs_fp64':reducer_relative,'deterministic':True,'tf32':False}),flush=True)
        assert average_check['relative_l2'] < 2e-5 and average_check['max_abs'] < 1e-6, average_check
        del full
    del gathered
    dist.barrier()
    torch.backends.cudnn.benchmark = backend['benchmark']
    torch.backends.cudnn.deterministic = backend['deterministic']
    torch.backends.cudnn.allow_tf32 = backend['cudnn_tf32']
    torch.backends.cuda.matmul.allow_tf32 = backend['matmul_tf32']
    fast = TrainingState(a, acceleration=True)
    for s in (ref, fast):
        s.scaler.set_growth_interval(2)
    rows = []
    gradient = None
    for i in range(12):
        batch = ref.inputs()
        lr = ref.update(batch, inject_inf=(i == 4))
        lf = fast.update(batch, inject_inf=(i == 4))
        assert ref.scaler.get_scale() == fast.scaler.get_scale()
        assert ref.skipped == fast.skipped
        row = {'step': i, 'reference_loss': float(lr), 'accelerated_loss': float(lf),
               'scale': fast.scaler.get_scale(), 'skips': fast.skipped}
        rows.append(row)
        if i == 0:
            gradient = diff([p.grad for p in fast.params], [p.grad for p in ref.params])
    checks = {'gradient': gradient, 'parameters': diff(fast.params, ref.params),
              'ema': diff(fast.ep, ref.ep), 'slow_ema': diff(fast.sp, ref.sp)}
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        checks[key] = diff([fast.opt.state[p][key] for p in fast.params], [ref.opt.state[p][key] for p in ref.params])
    loss_rel = max(abs(r['accelerated_loss']-r['reference_loss'])/max(abs(r['reference_loss']), 1e-20) for r in rows)
    passed = gradient['relative_l2'] < .005 and checks['parameters']['relative_l2'] < .0001 and checks['ema']['relative_l2'] < .0001 and checks['slow_ema']['relative_l2'] < .0001 and loss_rel < .005 and checks['step']['max_abs'] == 0 and checks['exp_avg']['relative_l2'] < .005 and checks['exp_avg_sq']['relative_l2'] < .005
    flag = torch.tensor(int(passed), device='cuda')
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    assert flag.item() == 1, checks
    replicas = fast.verify_replicas()
    if rank == 0:
        result = {'passed': True, 'kind': a.kind, 'world_size': world, 'global_batch': a.global_batch,
                  'gradient_average_vs_full_batch_fp32': average_check, 'checks': checks, 'rows': rows,
                  'max_loss_relative': loss_rel, 'replicas': replicas, 'reducer_vs_fp64_relative_l2': reducer_relative,
                  'limits_set_before_run': {'fp32_full_batch_relative_l2': 2e-5, 'fp32_full_batch_max_abs': 1e-6, 'reducer_vs_fp64_relative_l2': 1e-6, 'gradient_relative_l2': .005,
                    'parameters_and_emas_relative_l2': .0001, 'loss_relative': .005, 'optimizer_moments_relative_l2': .005},
                  'gate_revision': 'Initial low-LR UNet FP32 oracle failed at4.51e-5; next run failed at6.06e-4 despite added FP64 reducer check. Retain both failures. Fix oracle backend to deterministic, TF32 disabled; restore original2e-5 relative limit. Production AMP graph/eager tests restore original backend flags; optimizer/EMA thresholds unchanged.',
                  'scope': '12 paired full-state updates; overflow injected on rank0 only at step4; scaler growth interval2; gradient average before global clip; optimizer and both EMA checked.'}
        Path(a.out).write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
