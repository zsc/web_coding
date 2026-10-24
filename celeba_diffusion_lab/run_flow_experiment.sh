#!/bin/bash
set -euo pipefail
cd /home/zsc/celeba_diffusion_lab_20261024
python=/home/zsc/miniconda3/envs/py312/bin/python
# Separate sequential phase; do not contend with the affine timing benchmark.
while tmux has-session -t celeba-affine 2>/dev/null; do sleep 10; done
"$python" src/validate_flow.py > results/flow_validation.log 2>&1
"$python" src/train_flow.py --grayscale --variant jit --steps 20000 --batch 32 --out results/flow_jit20k > results/flow_jit20k.log 2>&1
"$python" src/train_flow.py --grayscale --variant unet --steps 5000 --batch 32 --out results/flow_unet5k > results/flow_unet5k.log 2>&1
"$python" src/evaluate_flow.py > results/flow_evaluate.log 2>&1
"$python" src/plot_flow.py > results/flow_plot.log 2>&1
