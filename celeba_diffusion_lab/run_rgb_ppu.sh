#!/bin/bash
set -euo pipefail
cd /dataset-cpfs5/vtm-foundation-pretrain/temp/zhousc6/celeba_diffusion_lab_20261024
export PYTHONPATH="$PWD/overlay"
py=/usr/local/bin/python3
"$py" - <<'PY'
import hashlib,json
from pathlib import Path
expected={'train':'8ab8f7934272871686400b5d109760c7b27970859015f6e86af4f374b08f1af7','valid':'03becc79ecc632efd762fd04f6f94d26fea4d29e05c8d146ac25456abf8ac9f9'}
record=json.loads(Path('data/nas_rgb_cache_verification.json').read_text())
for split,sha in expected.items():
    assert record[split]['sha256']==sha
    h=hashlib.sha256()
    with open('data/'+split+'.npy','rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    assert h.hexdigest()==sha
print('Exact RGB train/validation cache hashes verified',flush=True)
PY
"$py" -u src/validate_flow.py --channels 3 --out results/rgb_flow_validation.json > results/rgb_flow_validation.log 2>&1
"$py" -u src/train_flow.py --variant jit --steps 5000 --batch 32 --out results/rgb_flow_jit20k > results/rgb_flow_jit20k.log 2>&1
"$py" -u src/benchmark_device.py --channels 3 --label PPU-ZW810E --weights results/rgb_flow_jit20k/ema_5000.pt --out results/benchmark_rgb_ppu > results/benchmark_rgb_ppu.log 2>&1
"$py" -u src/train_flow.py --variant jit --steps 20000 --batch 32 --resume --out results/rgb_flow_jit20k >> results/rgb_flow_jit20k.log 2>&1
"$py" -u src/train_flow.py --variant unet --steps 5000 --batch 32 --out results/rgb_flow_unet5k > results/rgb_flow_unet5k.log 2>&1
