#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ssh desk 'cd mcmc_splatting_gpu_20261024 && ./desk_benchmark --fit --fp32 --chains 32 --gaussians 100 --steps 100000 --output results/fit'
scp -q 'desk:mcmc_splatting_gpu_20261024/results/fit.*' results/desk/
echo DONE
