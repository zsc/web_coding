#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && tail -8 results/bench_full.log; tail -6 results/validate_final.log 2>/dev/null' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/8eaa6047fbe140c6bc1aaf34261cabe7.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/8eaa6047fbe140c6bc1aaf34261cabe7.exit
