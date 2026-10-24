#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd ~/ppo_gpu_lab && /home/zsc/miniconda3/envs/py312/bin/python -u src/validate_v2.py' > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/3b914fcbfd614ea5b1db86894230f295.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/3b914fcbfd614ea5b1db86894230f295.exit
