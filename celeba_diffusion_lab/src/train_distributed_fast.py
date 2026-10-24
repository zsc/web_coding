"""RGB continuation fork on 1 node x16 PPUs; graph + averaged gradients."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from distributed_acceleration import init, TrainingState
from evaluate_distributed import evaluate


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=('unet','scaled'), required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--global-batch', type=int, default=512)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--target-images', type=int, default=5120000)
    p.add_argument('--seed', type=int, default=20261025)
    a = p.parse_args()
    rank, world = init()
    assert world == 16 and a.global_batch == 512
    root = Path(a.out)
    assert json.loads((root/'numerical_gate.json').read_text())['passed']
    if rank == 0:
        assert not (root/'training.json').exists(), 'Require a fresh experiment directory'
    s = TrainingState(a)
    assert (a.target_images-s.source_images) % a.global_batch == 0
    steps = (a.target_images-s.source_images)//a.global_batch
    assert steps > 0
    md = {'config': vars(a), 'world_size': world, 'local_batch': s.batch, 'channels': 3,
          'source_step': s.source_step, 'source_image_draws': s.source_images,
          'source_checkpoint_sha256': hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),
          'model': s.model.experiment_config, 'parameters': sum(p.numel() for p in s.params),
          'target_updates': steps, 'step': 0, 'training_image_draws': s.source_images,
          'source_successful_updates': s.source_successful, 'source_skips': s.source_skips,
          'acceleration': 'per-rank forward/backward graph; FP32 flat allreduce outside graph; Triton25 nonaffine RMSNorms for JiT',
          'replaced_norms': s.replaced_norms, 'graph_setup_seconds': s.graph.setup_seconds,
          'torch': torch.__version__, 'curve': [], 'status': 'training',
          'fork_note': 'Model, optimizer, scaler and EMA restored. Batch/world changed; new independent rank RNG streams. This is a batch512 recipe fork, not an exact batch32 continuation.',
          'ema': {'regular_per32_images': .999, 'slow_per32_images': .9999, 'decays_per_update': [.999**16,.9999**16],
                  'slow_origin': 'preserved from source' if s.source_has_slow else 'initialized from source regular EMA'},
          'recipe': 'RGB64, direct-x velocity MSE, logit-normal t(-.8,.8), hflip, unchanged normalization, AdamW wd1e-4, clip1 after average gradients, FP16 dynamic scaling, constant configured LR; no new warmup for full optimizer continuation.'}
    if rank == 0:
        write(root/'training.json', md)
        print(json.dumps({'event':'start','world':world,'source_images':s.source_images,'target_updates':steps,'config':vars(a)}), flush=True)
    s.model.train()
    dist.barrier()
    torch.cuda.synchronize()
    start = block = time.perf_counter()
    values = []
    for step in range(1, steps+1):
        values.append(s.update())
        if step % 100 == 0 or step == steps:
            torch.cuda.synchronize()
            seconds = time.perf_counter()-block
            stats = torch.stack([torch.stack(values).mean(), torch.tensor(seconds, device='cuda')])
            dist.all_reduce(stats[:1], op=dist.ReduceOp.SUM)
            dist.all_reduce(stats[1:], op=dist.ReduceOp.MAX)
            loss, elapsed = stats.cpu().tolist()
            loss /= world
            assert np.isfinite(loss)
            row = {'step':step,'training_image_draws':s.source_images+step*a.global_batch,
                   'loss':loss,'ms_per_step':elapsed*1000/len(values),'images_per_second':len(values)*a.global_batch/elapsed,
                   'successful_updates':s.successful,'skipped':s.skipped,'scale':s.scaler.get_scale(),
                   'elapsed_seconds':time.perf_counter()-start}
            if rank == 0:
                md['curve'].append(row)
                md.update(row)
                write(root/'training.json', md)
                print(json.dumps(row), flush=True)
            values = []
            block = time.perf_counter()
        images = s.source_images+step*a.global_batch
        quality_due = images % 1280000 == 0 or step == steps
        if step % 1250 == 0 or step == steps:
            pause = time.perf_counter()
            replica = s.verify_replicas()
            cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
            checkpoint_dir = root/f'checkpoint_{step}'
            if rank == 0:
                checkpoint_dir.mkdir()
            dist.barrier()
            torch.save({'rank':rank,'step':step,'cpu_rng':cpu,'cuda_rng':cuda,'numpy_rng':s.rng.bit_generator.state}, checkpoint_dir/f'rng_rank{rank}.pt')
            if rank == 0:
                base = {'step':step,'source_step':s.source_step,'config':vars(a),'world_size':world,
                        'model_config':s.model.experiment_config,'training_image_draws':images}
                ckpt = {**base,'model':s.model.state_dict(),'ema':s.ema.state_dict(),'slow_ema':s.slow.state_dict(),
                        'optimizer':s.opt.state_dict(),'scaler':s.scaler.state_dict(),
                        'successful':s.source_successful+s.successful,'skipped':s.source_skips+s.skipped}
                torch.save(ckpt, checkpoint_dir/'state.pt')
                if quality_due:
                    torch.save({**base,'ema':s.ema.state_dict()}, root/f'ema_{step}.pt')
                    torch.save({**base,'ema':s.slow.state_dict()}, root/f'slow_ema_{step}.pt')
                md.update(step=step,training_image_draws=images,status='evaluating' if quality_due else 'training',replicas=replica)
                write(root/'training.json', md)
                write(root/f'training_{step}.json', md)
                print(json.dumps({'event':'checkpoint','step':step,'images':images,'replicas':replica}),flush=True)
            dist.barrier()
            if quality_due:
                evaluate(root, step, a.kind)
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(cuda)
            dist.barrier()
            block += time.perf_counter()-pause
            if rank == 0:
                md['status'] = 'complete' if step == steps else 'training'
                write(root/'training.json', md)
    if rank == 0:
        print(json.dumps({'event':'complete','step':steps,'images':a.target_images}),flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
