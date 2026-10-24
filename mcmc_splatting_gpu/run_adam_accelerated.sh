#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/adam_fast
scp -q src/graph_fit.cu src/roi_fit.cu src/desk_benchmark.cu desk:mcmc_splatting_gpu_20261024/src/
ssh desk 'cd mcmc_splatting_gpu_20261024 && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=true -DADAM_ROI_SIGMA=4.5f -Xcompiler -fopenmp src/graph_fit.cu -o adam_roi_4p5 && ./adam_roi_4p5 --mode 0 --warp-roi 1 --verify-threads 12 --steps 8000 --output results/adam_fast'
scp -q 'desk:mcmc_splatting_gpu_20261024/results/adam_fast.*' results/adam_fast/
echo COPIED
