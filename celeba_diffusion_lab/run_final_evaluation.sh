#!/bin/bash
set -euo pipefail
cd /home/zsc/celeba_diffusion_lab_20261024
python=/home/zsc/miniconda3/envs/py312/bin/python
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
while tmux has-session -t celeba-gray 2>/dev/null; do sleep 15; done
"$python" - <<'PY'
import json
from pathlib import Path
t=json.loads(Path('results/gray5k/training.json').read_text())
assert t['status']=='complete' and t['step']==5000 and t['config']['grayscale']
PY
"$python" src/evaluate_gray.py --samples 512 > results/gray_evaluate.log 2>&1
"$python" src/evaluate.py --kid > results/evaluate.log 2>&1
printf 'All evaluations completed.\n'
