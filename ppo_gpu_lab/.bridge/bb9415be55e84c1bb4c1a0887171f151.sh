#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd ~/ppo_gpu_lab && /home/zsc/miniconda3/envs/py312/bin/python src/diagnose.py && /home/zsc/miniconda3/envs/py312/bin/python -u src/run.py --mode cpu --envs 256 --horizon 64 --minibatch 2048 --iterations 7 --out cpu_scaled' > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/bb9415be55e84c1bb4c1a0887171f151.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/bb9415be55e84c1bb4c1a0887171f151.exit
