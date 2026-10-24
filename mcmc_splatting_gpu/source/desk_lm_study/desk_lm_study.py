#!/usr/bin/env python3
"""Run on desk: process timers, fixed time-calibrated budgets, paired seeds."""
import hashlib
import json
import pathlib
import subprocess
import time

root = pathlib.Path.cwd()
out = root / 'results/lm_study'
out.mkdir(parents=True, exist_ok=True)
variants = [('adam', 8000, 0), ('block_lm', 0, 3456), ('adam_then_lm', 1000, 3120)]
manifest = {
    'variants': variants,
    'time_calibration': {
        'baseline_process_ms': 3720, 'pure_lm_1000_ms': 2070, 'pure_lm_3400_ms': 3680,
        'warm1000_lm2900_ms': 3570, 'estimated_lm_ms_per_step': (3680-2070)/(3400-1000),
        'rule': 'Interpolate total wall time to baseline, round LM steps to nearest 32. For warm start use the measured pure-LM marginal slope. Freeze budgets before repeats and holdout seeds. No MSE enters calibration.'},
    'sources': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in [
        'src/lm_fit.cu', 'src/roi_fit.cu', 'src/desk_benchmark.cu', 'src/desk_lm_study.py']},
    'target_sha256': hashlib.sha256((root/'zhou_target.pgm').read_bytes()).hexdigest(),
    'timing': 'Python perf_counter around subprocess.run; includes launch, CUDA context, initialization, optimization, readback, full FP64 rendering, file output, cleanup and process exit. Excludes compilation, SSH and this harness setup.',
    'runs': [], 'complete': False}
manifest['hardware_before'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version', '--format=csv,noheader'], text=True).strip()

def run(variant, seed, phase, repeat):
    name, adam, lm = variant
    stem = f'{phase}_{repeat}_{name}_seed{seed}'
    cmd = ['./lm_fit', '--input', 'zhou_target.pgm', '--gaussians', '1600', '--chains', '8',
           '--adam-steps', str(adam), '--lm-steps', str(lm), '--seed', str(seed), '--output', str(out/stem)]
    t = time.perf_counter()
    process = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall_ms = (time.perf_counter()-t)*1000
    r = json.loads((out/(stem+'.json')).read_text())
    record = {'file': stem+'.json', 'group': name, 'phase': phase, 'repeat': repeat, 'seed': seed,
              'wall_ms': wall_ms, 'gpu_ms': r['gpu_ms'], 'loss': r['loss'], 'smooth_loss': r['smooth_loss'],
              'adam_steps': adam, 'lm_steps': lm, 'stdout': process.stdout.strip(), 'command': cmd}
    manifest['runs'].append(record)
    (out/'study.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('command','stdout')}), flush=True)

run(variants[0], 42, 'warmup', 0)
for repeat in range(3):
    for i in range(3):
        run(variants[(i+repeat)%3], 42, 'repeat', repeat)
# Budgets are frozen; fresh seed ensembles are not used for selecting settings.
for repeat, seed in enumerate([1042,2042,3042]):
    for i in range(3):
        run(variants[(i+repeat)%3], seed, 'holdout', repeat)
manifest['hardware_after'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version', '--format=csv,noheader'], text=True).strip()
manifest['complete'] = True
(out/'study.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('DONE', flush=True)
