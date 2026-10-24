#!/bin/bash
set -euo pipefail
cd /home/zsc/celeba_diffusion_lab_20261024
python=/home/zsc/miniconda3/envs/py312/bin/python
for variant in time_affine global_affine; do
  "$python" src/train.py --grayscale --variant "$variant" --steps 5000 --batch 32 --out "results/${variant}5k" > "results/${variant}5k.log" 2>&1
done
"$python" src/evaluate_affine.py > results/affine_evaluate.log 2>&1
"$python" src/plot_affine.py > results/affine_plot.log 2>&1
