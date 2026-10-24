#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd ~/ppo_gpu_lab && /home/zsc/miniconda3/envs/py312/bin/python -u src/benchmark_v2.py' > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/347b7f0ff8c1498fb56ebf1769fdaddb.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/347b7f0ff8c1498fb56ebf1769fdaddb.exit
