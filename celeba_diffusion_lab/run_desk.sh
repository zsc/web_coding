#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
remote=/home/zsc/celeba_diffusion_lab_20261024
python=/home/zsc/miniconda3/envs/py312/bin/python
control=${DESK_CONTROL_PATH:-/tmp/mnist-desk-control}
case "${1:-status}" in
  sync)
    rsync -az -e "ssh -S $control" src/ "desk:$remote/src/"
    ;;
  status)
    ssh -S "$control" desk "tail -n 2 $remote/results/gray5k.log; tail -n 2 $remote/results/gray_evaluate.log 2>/dev/null; tail -n 2 $remote/results/evaluation_pipeline.log 2>/dev/null; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv"
    ;;
  train)
    ssh -S "$control" desk "tmux new-session -d -s celeba-diffusion 'cd $remote && $python src/train.py --steps 10000 --batch 32 > results/train.log 2>&1'"
    ;;
  resume)
    steps=${2:-20000}
    [[ "$steps" =~ ^[0-9]+$ ]]
    ssh -S "$control" desk "tmux new-session -d -s celeba-diffusion 'cd $remote && $python src/train.py --resume --steps $steps --batch 32 >> results/train.log 2>&1'"
    ;;
  evaluate)
    ssh -S "$control" desk "cd $remote && $python src/evaluate.py --kid > results/evaluate.log 2>&1"
    ;;
  gray)
    ssh -S "$control" desk "tmux new-session -d -s celeba-gray 'cd $remote && $python src/train.py --grayscale --steps 5000 --batch 32 --out results/gray5k > results/gray5k.log 2>&1'"
    ;;
  affine)
    scp -o "ControlPath=$control" run_affine_experiment.sh "desk:$remote/"
    ssh -S "$control" desk "tmux new-session -d -s celeba-affine 'bash $remote/run_affine_experiment.sh > $remote/results/affine_pipeline.log 2>&1'"
    ;;
  affine-status)
    ssh -S "$control" desk "tail -n 2 $remote/results/time_affine5k.log; tail -n 2 $remote/results/global_affine5k.log 2>/dev/null; tail -n 2 $remote/results/affine_evaluate.log 2>/dev/null; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv"
    ;;
  gray-evaluate)
    ssh -S "$control" desk "cd $remote && $python src/evaluate_gray.py --samples 512 > results/gray_evaluate.log 2>&1"
    ;;
  fetch)
    rsync -az --exclude='*.pt' --exclude='*.npy' -e "ssh -S $control" "desk:$remote/results/" results/
    scp -o "ControlPath=$control" "desk:$remote/data/manifest.json" "desk:$remote/data/gray_manifest.json" data/
    ;;
  *) echo 'Usage: run_desk.sh {sync|status|train|resume [total_steps]|evaluate|gray|gray-evaluate|affine|affine-status|fetch}' >&2; exit 1 ;;
esac
