"""Record the actual RGB experiment code, weights and runtime without old gray runs."""
import datetime
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import torch
from model import gpu_status


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


p = argparse.ArgumentParser()
p.add_argument('--out', default='results/rgb_flow_environment.json')
args = p.parse_args()
files = list(Path('src').rglob('*.py')) + [Path('run_rgb_ppu.sh'), Path('data/manifest.json'),
    Path('data/nas_rgb_cache_verification.json')]
for name in ('rgb_flow_jit20k', 'rgb_flow_unet5k', 'rgb_scaled_jit10k', 'rgb_flow_unet40k'):
    files += list((Path('results')/name).glob('ema_*.pt'))
    files += list((Path('results')/name).glob('training*.json'))
for name in ('rgb_flow_evaluation', 'rgb_scaled_evaluation', 'benchmark_rgb_ppu', 'rgb_unet_continuation_evaluation'):
    files += list((Path('results')/name).glob('*.json'))
files += list(Path('results').glob('scaled_bench_*ppu/benchmark.json'))
files += list(Path('results').glob('unet_scaling_*ppu/benchmark.json'))
r = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'python': sys.version, 'platform': platform.platform(), 'torch_cuda': torch.version.cuda,
     'packages': {p: importlib.metadata.version(p) for p in ('torch','torchvision','diffusers','numpy','pillow','pytorch-fid')},
     'gpu': gpu_status(), 'files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files]}
Path(args.out).write_text(json.dumps(r, indent=2))
print('Recorded RGB sources, EMA weights and runtime.')
