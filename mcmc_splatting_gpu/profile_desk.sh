#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Read-only counter collection. Does not change driver counter policy, clocks or cache state.
# On this desk, root is needed for GPU counters; -n fails instead of prompting.
ssh desk 'cd mcmc_splatting_gpu_20261024 && sudo -n /opt/nvidia/nsight-compute/2024.1.1/ncu --target-processes all --kernel-name regex:iterateKernel --launch-skip 195 --launch-count 1 --section SpeedOfLight --section Occupancy --section SchedulerStats --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second --clock-control none --cache-control none --csv ./desk_benchmark --fit --fp32 --chains 32 --gaussians 100 --steps 100000 --output results/profile_privileged > results/ncu_privileged.csv 2> results/ncu_privileged.log'
scp -q 'desk:mcmc_splatting_gpu_20261024/results/ncu_privileged.*' results/desk/
