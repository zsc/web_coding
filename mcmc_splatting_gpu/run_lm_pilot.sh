#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/desk_lm
scp -q src/desk_benchmark.cu src/roi_fit.cu src/lm_fit.cu desk:mcmc_splatting_gpu_20261024/src/
ssh desk 'cd mcmc_splatting_gpu_20261024 && mkdir -p results/lm && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false -Xptxas=-v src/lm_fit.cu -o lm_fit'
ssh desk 'cd mcmc_splatting_gpu_20261024 && ./lm_fit --gaussians 32 --chains 1 --adam-steps 0 --lm-steps 1 --audit results/lm/audit.json --output results/lm/check'
for spec in '0 100' '0 1000' '500 1000' '8000 0'; do
 read -r adam lm <<< "$spec"
 ssh desk "cd mcmc_splatting_gpu_20261024 && /usr/bin/time -f 'process_wall_seconds=%e' -o results/lm/pilot_${adam}_${lm}.time ./lm_fit --adam-steps $adam --lm-steps $lm --output results/lm/pilot_${adam}_${lm}"
done
scp -q 'desk:mcmc_splatting_gpu_20261024/results/lm/*' results/desk_lm/
echo DONE
