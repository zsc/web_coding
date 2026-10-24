#!/bin/sh
scp -r desk:/home/zsc/ppo_gpu_lab/results/phase_profile_v1.json ppo_gpu_lab/results/phase_profile_v1.json > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/e5f138f10eda4e6ba82f7dec63314740.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/e5f138f10eda4e6ba82f7dec63314740.exit
