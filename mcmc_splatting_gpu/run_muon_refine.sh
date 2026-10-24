#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/desk_muon_refine source/desk_muon_refine
cp src/muon_fit.cu src/roi_fit.cu src/desk_benchmark.cu src/desk_muon_refine.py source/desk_muon_refine/
scp -q src/muon_fit.cu src/desk_muon_refine.py desk:mcmc_splatting_gpu_20261024/src/
ssh desk 'cd mcmc_splatting_gpu_20261024 && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false src/muon_fit.cu -o muon_fit && python3 src/desk_muon_refine.py'
scp -q 'desk:mcmc_splatting_gpu_20261024/results/muon_refine/*' results/desk_muon_refine/
echo COPIED
