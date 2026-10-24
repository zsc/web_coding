#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/desk_graph_study source/desk_graph_study
cp src/graph_fit.cu src/roi_fit.cu src/desk_benchmark.cu src/desk_graph_study.py source/desk_graph_study/
scp -q src/graph_fit.cu src/roi_fit.cu src/desk_benchmark.cu src/desk_graph_study.py desk:mcmc_splatting_gpu_20261024/src/
ssh desk 'cd mcmc_splatting_gpu_20261024 && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false -Xcompiler -fopenmp src/graph_fit.cu -o graph_fit && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=true -Xcompiler -fopenmp src/graph_fit.cu -o graph_fit_fma && python3 src/desk_graph_study.py'
scp -q 'desk:mcmc_splatting_gpu_20261024/results/graph_study/*' results/desk_graph_study/
echo COPIED
