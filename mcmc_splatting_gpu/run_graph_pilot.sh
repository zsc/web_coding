#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/desk_graph_pilot source/desk_graph_pilot
cp src/graph_fit.cu src/roi_fit.cu src/desk_benchmark.cu src/desk_graph_pilot.py source/desk_graph_pilot/
scp -q src/graph_fit.cu src/desk_graph_pilot.py desk:mcmc_splatting_gpu_20261024/src/
ssh desk 'cd mcmc_splatting_gpu_20261024 && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false -Xcompiler -fopenmp src/graph_fit.cu -o graph_fit && python3 src/desk_graph_pilot.py'
scp -q 'desk:mcmc_splatting_gpu_20261024/results/graph_pilot/*' results/desk_graph_pilot/
echo COPIED
