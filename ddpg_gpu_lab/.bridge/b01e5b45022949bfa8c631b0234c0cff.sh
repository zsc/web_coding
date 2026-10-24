#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader; cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && ls results/train_extended*.json' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/b01e5b45022949bfa8c631b0234c0cff.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/b01e5b45022949bfa8c631b0234c0cff.exit
