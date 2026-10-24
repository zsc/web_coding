#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd ~/ppo_gpu_lab && /home/zsc/miniconda3/envs/py312/bin/python -u src/profile_v1.py' > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/4b4a54ffb112498ebcd481c818bd103a.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/4b4a54ffb112498ebcd481c818bd103a.exit
