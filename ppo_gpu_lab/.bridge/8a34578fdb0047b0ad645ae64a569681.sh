#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd ~/ppo_gpu_lab && /home/zsc/miniconda3/envs/py312/bin/python -u src/run.py --mode eager --envs 256 --horizon 64 --minibatch 2048 --iterations 7 --out eager_scaled' > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/8a34578fdb0047b0ad645ae64a569681.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/8a34578fdb0047b0ad645ae64a569681.exit
