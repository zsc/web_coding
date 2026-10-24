#!/bin/sh
scp -r desk:/home/zsc/ppo_gpu_lab/results/cpu_matched.json ppo_gpu_lab/results/desk/cpu_matched.json > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/9266ec7dc3434ce292f88dbf70f59f81.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/9266ec7dc3434ce292f88dbf70f59f81.exit
