#!/bin/sh
# Re-run the validated accelerated workload using the existing desk environment.
set -eu
cd "$(dirname "$0")"
python3 bridge.py run 'mkdir -p ~/ppo_gpu_lab/src ~/ppo_gpu_lab/source ~/ppo_gpu_lab/results'
python3 bridge.py push src /home/zsc/ppo_gpu_lab/
python3 bridge.py push source/original.html /home/zsc/ppo_gpu_lab/source/
python3 bridge.py run 'cd ~/ppo_gpu_lab && /home/zsc/.nvm/versions/node/v24.13.1/bin/node src/baseline.cjs && /home/zsc/miniconda3/envs/py312/bin/python src/validate.py && /home/zsc/miniconda3/envs/py312/bin/python src/graph_check.py && /home/zsc/miniconda3/envs/py312/bin/python src/run.py --mode v2 --envs 256 --horizon 64 --minibatch 2048 --iterations 300 --physical --evaluate --out rerun'
python3 bridge.py pull /home/zsc/ppo_gpu_lab/results/rerun.json results/rerun.json
python3 bridge.py pull /home/zsc/ppo_gpu_lab/results/rerun.pt results/rerun.pt
