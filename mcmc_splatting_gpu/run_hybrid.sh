#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
scp -q src/roi_fit.cu desk:mcmc_splatting_gpu_20261024/src/
ssh desk 'cd mcmc_splatting_gpu_20261024 && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false src/roi_fit.cu -o hybrid_fit'
for spec in '0 500' '10000 500' '10000 1500'; do
  read -r steps adam <<< "$spec"
  ssh desk "cd mcmc_splatting_gpu_20261024 && ./hybrid_fit --steps $steps --adam-steps $adam --output results/hybrid_${steps}_${adam}"
  scp -q "desk:mcmc_splatting_gpu_20261024/results/hybrid_${steps}_${adam}.*" results/desk/
done
