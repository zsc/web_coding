#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
scp -q src/desk_benchmark.cu src/roi_fit.cu desk:mcmc_splatting_gpu_20261024/src/
scp -q assets/zhou_target.pgm desk:mcmc_splatting_gpu_20261024/
ssh desk 'cd mcmc_splatting_gpu_20261024 && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false src/roi_fit.cu -o hybrid_fit'
# Measured process time begins after compilation, excludes network copy and report generation.
ssh desk 'cd mcmc_splatting_gpu_20261024 && /usr/bin/time -f "process_wall_seconds=%e" -o results/fast.time ./hybrid_fit --input zhou_target.pgm --steps 10000 --adam-steps 8000 --gaussians 1600 --chains 8 --output results/fast'
scp -q 'desk:mcmc_splatting_gpu_20261024/results/fast.*' results/desk/
