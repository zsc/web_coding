#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
REMOTE_DIR=${DIFFUSION_REMOTE_DIR:-/home/zsc/diffusion_gpu_lab_20261024}
REMOTE_PYTHON=${DIFFUSION_REMOTE_PYTHON:-/home/zsc/miniconda3/envs/py312/bin/python}
SSH=(ssh)
if [[ -S /tmp/mnist-desk-control ]]; then SSH+=(-S /tmp/mnist-desk-control); fi
# Fixed known paths are deliberate; avoid interpolating arbitrary shell fragments.
if [[ "$REMOTE_DIR" != /home/zsc/diffusion_gpu_lab_20261024 || "$REMOTE_PYTHON" != /home/zsc/miniconda3/envs/py312/bin/python ]]; then
  echo 'Edit this script for a different remote layout.' >&2; exit 2
fi
case "${1:-sample}" in
  sync)
    "${SSH[@]}" desk "mkdir -p $REMOTE_DIR/results"
    rsync -az -e "${SSH[*]}" --exclude '__pycache__' --exclude results --exclude report.html ./ "desk:$REMOTE_DIR/"
    rsync -az -e "${SSH[*]}" results/checkpoint_epoch5.pt results/checkpoint_epoch20.pt "desk:$REMOTE_DIR/results/" ;;
  train) "${SSH[@]}" desk "cd $REMOTE_DIR && $REMOTE_PYTHON -u src/train.py --epochs 20 --mode graph --out results/graph_training" ;;
  benchmark) "${SSH[@]}" desk "cd $REMOTE_DIR && $REMOTE_PYTHON -u src/benchmark.py && $REMOTE_PYTHON -u src/verify_more.py && $REMOTE_PYTHON -u src/validate_training.py && $REMOTE_PYTHON -u src/train_ablation.py" ;;
  sample) "${SSH[@]}" desk "cd $REMOTE_DIR && $REMOTE_PYTHON src/sample.py" ;;
  fetch) rsync -az -e "${SSH[*]}" "desk:$REMOTE_DIR/results/" results/ ;;
  *) echo 'Usage: ./run_desk.sh {sync|train|benchmark|sample|fetch}' >&2; exit 2 ;;
esac
