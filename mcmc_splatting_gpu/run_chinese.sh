#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
scp -q desk:mcmc_splatting_gpu_20261024/src/desk_benchmark.cu source/desk_benchmark.first_suite.cu
scp -q src/desk_benchmark.cu desk:mcmc_splatting_gpu_20261024/src/
scp -q assets/zhou_target.pgm desk:mcmc_splatting_gpu_20261024/
ssh desk 'cd mcmc_splatting_gpu_20261024 && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false src/desk_benchmark.cu -o desk_benchmark'
for steps in 100000 500000; do
  ssh desk "cd mcmc_splatting_gpu_20261024 && ./desk_benchmark --fit --fp32 --seeded --input zhou_target.pgm --chains 8 --gaussians 800 --steps $steps --initial-temperature 2 --final-temperature 0.001 --output results/zhou_$steps"
  scp -q "desk:mcmc_splatting_gpu_20261024/results/zhou_$steps.*" results/desk/
done
echo DONE
