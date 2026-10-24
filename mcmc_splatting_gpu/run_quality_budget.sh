#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for spec in '800 5000' '1200 6000' '1600 8000'; do
  read -r gs adam <<< "$spec"
  ssh desk "cd mcmc_splatting_gpu_20261024 && /usr/bin/time -f 'process_wall_seconds=%e' -o results/budget_${gs}_${adam}.time ./hybrid_fit --steps 10000 --adam-steps $adam --gaussians $gs --output results/budget_${gs}_${adam}"
  scp -q "desk:mcmc_splatting_gpu_20261024/results/budget_${gs}_${adam}.*" results/desk/
done
