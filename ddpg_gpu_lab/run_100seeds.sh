#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${DDPG_PYTHON:-/home/zsc/miniconda3/envs/py312/bin/python}
NODE=${DDPG_NODE:-/home/zsc/.nvm/versions/node/v24.13.1/bin/node}
"$PYTHON" validate_td3.py --cuda
"$PYTHON" validate_symmetry.py
/usr/bin/time -f '{"process_wall_seconds":%e}' -o results/seeds100_td3_process_time.json "$PYTHON" multiseed.py --algo td3 --symmetry --seeds 100 --seed-start 1 \
  --envs 32 --batch 128 --updates 200000 --eval-every 10000 \
  --episode-limit 2000 --horizon 100000 --eval-episodes 64 \
  --lr-actor 0.0003 --lr-critic 0.0003 --freeze-passed \
  --out results/seeds100_td3
cp results/seeds100_td3/summary.json results/seeds100_td3.json
"$PYTHON" verify_policies.py results/seeds100_td3.json --horizon 100000 --episodes 256 --split 997 --out results/verification_td3.json
"$NODE" verify_original.cjs results/seeds100_td3.json 10000 results/verification_original.json
