#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd ~/ppo_gpu_lab && OMP_NUM_THREADS=1 /home/zsc/miniconda3/envs/py312/bin/python -u src/run.py --mode cpu --envs 64 --horizon 32 --minibatch 64 --iterations 5 --out cpu_matched' > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/63a191f01287447ea4263e6fb4644f2c.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/63a191f01287447ea4263e6fb4644f2c.exit
