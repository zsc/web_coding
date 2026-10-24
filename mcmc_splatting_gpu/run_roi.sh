#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
scp -q src/roi_fit.cu desk:mcmc_splatting_gpu_20261024/src/
ssh desk 'cd mcmc_splatting_gpu_20261024 && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false src/roi_fit.cu -o roi_fit'
for steps in 500000 1000000 2000000; do
  ssh desk "cd mcmc_splatting_gpu_20261024 && ./roi_fit --steps $steps --output results/roi_$steps"
  scp -q "desk:mcmc_splatting_gpu_20261024/results/roi_$steps.*" results/desk/
done
echo DONE
