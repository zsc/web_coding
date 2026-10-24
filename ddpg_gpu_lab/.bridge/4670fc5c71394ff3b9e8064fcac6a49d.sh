#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && tail -5 results/bench_full.log; tail -6 results/validate_final.log; for f in results/train_gpu_seed*.log; do echo "$f"; tail -3 "$f"; done' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/4670fc5c71394ff3b9e8064fcac6a49d.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/4670fc5c71394ff3b9e8064fcac6a49d.exit
