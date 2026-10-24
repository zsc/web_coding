#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/desk_muon_study source/desk_muon_study
cp src/muon_fit.cu src/roi_fit.cu src/desk_benchmark.cu src/desk_muon_study.py source/desk_muon_study/
scp -q src/desk_benchmark.cu src/roi_fit.cu src/muon_fit.cu src/desk_muon_study.py desk:mcmc_splatting_gpu_20261024/src/
ssh desk 'cd mcmc_splatting_gpu_20261024 && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false src/muon_fit.cu -o muon_fit && python3 src/desk_muon_study.py'
scp -q 'desk:mcmc_splatting_gpu_20261024/results/muon_study/*' results/desk_muon_study/
echo COPIED
