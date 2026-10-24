#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for rep in 1 2 3; do
  ssh desk "cd mcmc_splatting_gpu_20261024 && /usr/bin/time -f 'process_wall_seconds=%e' -o results/final_repeat_${rep}.time ./hybrid_fit --steps 10000 --adam-steps 8000 --gaussians 1600 --chains 8 --output results/final_repeat_${rep}"
  scp -q "desk:mcmc_splatting_gpu_20261024/results/final_repeat_${rep}.*" results/desk/
done
ssh desk 'cd mcmc_splatting_gpu_20261024 && sudo -n /opt/nvidia/nsight-compute/2024.1.1/ncu --target-processes all --kernel-name "regex:renderSparse|refineAdam" --launch-skip 8000 --launch-count 2 --section SpeedOfLight --section Occupancy --section SchedulerStats --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second --clock-control none --cache-control none --csv ./hybrid_fit --steps 10000 --adam-steps 8000 --gaussians 1600 --chains 8 --output results/profile_final > results/ncu_final.csv 2> results/ncu_final.log'
scp -q 'desk:mcmc_splatting_gpu_20261024/results/ncu_final.*' results/desk/
echo DONE
