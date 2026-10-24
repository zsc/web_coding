#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/desk_roi_tight_pilot source/desk_roi_tight_pilot
cp src/graph_fit.cu src/roi_fit.cu src/desk_benchmark.cu src/desk_roi_pilot.py source/desk_roi_tight_pilot/
scp -q src/graph_fit.cu src/roi_fit.cu src/desk_roi_pilot.py desk:mcmc_splatting_gpu_20261024/src/
ssh desk 'cd mcmc_splatting_gpu_20261024 && for sigma in 6 5 4.5 4 3.5; do name=${sigma/./p}; value=$sigma; if [[ $sigma != *.* ]]; then value=${sigma}.0; fi; /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=true -DADAM_ROI_SIGMA=${value}f -Xcompiler -fopenmp src/graph_fit.cu -o adam_roi_${name} || exit; done && python3 src/desk_roi_pilot.py'
scp -q 'desk:mcmc_splatting_gpu_20261024/results/roi_tight_pilot/*' results/desk_roi_tight_pilot/
echo COPIED
