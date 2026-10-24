#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/desk_roi_tight_study source/desk_roi_tight_study
cp src/graph_fit.cu src/roi_fit.cu src/desk_benchmark.cu src/desk_roi_study.py source/desk_roi_tight_study/
scp -q src/desk_roi_study.py desk:mcmc_splatting_gpu_20261024/src/
ssh desk 'cd mcmc_splatting_gpu_20261024 && python3 src/desk_roi_study.py'
scp -q 'desk:mcmc_splatting_gpu_20261024/results/roi_tight_study/*' results/desk_roi_tight_study/
echo COPIED
