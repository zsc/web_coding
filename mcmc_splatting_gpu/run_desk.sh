#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
remote="${MCMC_REMOTE_DIR:-mcmc_splatting_gpu_20261024}"
[[ "$remote" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Remote directory must be a simple name'; exit 2; }
mkdir -p results/desk
ssh -o BatchMode=yes -o ConnectTimeout=10 desk "mkdir -p '$remote/src' '$remote/results'; nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu --format=csv" > results/desk/before.txt
scp -q src/desk_benchmark.cu "desk:$remote/src/"
ssh desk "cd '$remote' && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false src/desk_benchmark.cu -o desk_benchmark" 2>&1 | tee results/desk/build.log
for spec in 'single_fp64 1' 'batch_fp64 32' 'single_fp32 1 --fp32' 'batch_fp32 32 --fp32'; do
  read -r name chains flag <<< "$spec"
  ssh desk "cd '$remote' && ./desk_benchmark --chains '$chains' --steps 1200 --repeats 3 ${flag:-} > 'results/$name.json' 2> 'results/$name.log'" || { scp -q "desk:$remote/results/*" results/desk/; exit 1; }
  scp -q "desk:$remote/results/$name.*" results/desk/
done
ssh desk 'nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv' > results/desk/after.txt
echo 'DONE'
