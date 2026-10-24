#!/bin/bash
set -euo pipefail
# Run on desk from /home/zsc/celeba_diffusion_lab_20261024 after the 10k baseline.
python=/home/zsc/miniconda3/envs/py312/bin/python
"$python" - <<'PY'
import json
from pathlib import Path
t=json.loads(Path('results/run1/training.json').read_text())
assert t['step']==10000 and t['status']=='complete',t['step']
PY
for name in control12k dc12k; do
  mkdir "results/$name"
  cp results/run1/checkpoint.pt "results/$name/checkpoint.pt"
  cp results/run1/training.json "results/$name/training.json"
done
"$python" src/train.py --resume --steps 12000 --batch 32 --out results/control12k > results/control12k.log 2>&1
"$python" src/train.py --resume --steps 12000 --batch 32 --out results/dc12k --dc-weight 256 --dc-t-min 800 > results/dc12k.log 2>&1
"$python" src/diagnose_color.py --checkpoint results/control12k/checkpoint.pt --out results/control_color --samples 64 > results/control_color.log 2>&1
"$python" src/diagnose_color.py --checkpoint results/dc12k/checkpoint.pt --out results/dc_color --samples 64 > results/dc_color.log 2>&1
