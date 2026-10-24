#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && tail -5 results/bench.log; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/749eab1991104488b93d86fd62cbd9d7.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/749eab1991104488b93d86fd62cbd9d7.exit
