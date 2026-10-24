#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && tail -12 results/validate_multiseed.log; tail -8 results/seeds100_ddpg_long.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/42325f9f5a494bbbaf6c313668144d7e.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/42325f9f5a494bbbaf6c313668144d7e.exit
