#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/desk_lm_study
scp -q src/desk_benchmark.cu src/roi_fit.cu src/lm_fit.cu src/desk_lm_study.py desk:mcmc_splatting_gpu_20261024/src/
ssh desk 'cd mcmc_splatting_gpu_20261024 && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false src/lm_fit.cu -o lm_fit && python3 src/desk_lm_study.py'
scp -q 'desk:mcmc_splatting_gpu_20261024/results/lm_study/*' results/desk_lm_study/
echo COPIED
