#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/desk_muon_pilot source/desk_muon_pilot
cp src/muon_fit.cu src/roi_fit.cu src/desk_benchmark.cu src/desk_muon_pilot.py source/desk_muon_pilot/
scp -q src/desk_benchmark.cu src/roi_fit.cu src/muon_fit.cu src/desk_muon_pilot.py desk:mcmc_splatting_gpu_20261024/src/
ssh desk 'cd mcmc_splatting_gpu_20261024 && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false src/muon_fit.cu -o muon_fit && python3 src/desk_muon_pilot.py'
scp -q 'desk:mcmc_splatting_gpu_20261024/results/muon_pilot/*' results/desk_muon_pilot/
echo COPIED
