#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${DDPG_PYTHON:-/home/zsc/miniconda3/envs/py312/bin/python}
NODE=${DDPG_NODE:-/home/zsc/.nvm/versions/node/v24.13.1/bin/node}
mkdir -p results
"$PYTHON" extract.py
"$NODE" baseline.cjs --mode=fixture --out=results/fixture.json
"$PYTHON" validate.py --cuda
"$NODE" baseline.cjs --mode=bench --out=results/desk_baseline.json
"$PYTHON" ddpg.py bench --updates 300 --out results/desk_benchmark.json
"$PYTHON" bench_full.py
for seed in 1 2 3; do
  "$PYTHON" ddpg.py train --device cuda --compile-full --graph --envs 32 --batch 256 --updates 20000 --seed "$seed" --out "results/train_gpu_seed${seed}.json"
done
"$NODE" baseline.cjs --mode=train --steps=20000 --seed=1 --out=results/train_original_seed1.json
