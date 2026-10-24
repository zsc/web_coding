#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd ~/ppo_gpu_lab && /home/zsc/miniconda3/envs/py312/bin/python -u src/run.py --mode cpu --envs 64 --horizon 32 --minibatch 64 --iterations 7 --out cpu_matched && nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv' > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/c8512dc9edf04fdcb863a51396390a04.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/c8512dc9edf04fdcb863a51396390a04.exit
