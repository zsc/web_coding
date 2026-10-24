#!/bin/sh
scp -r desk:/home/zsc/ppo_gpu_lab/results/eager_scaled.json ppo_gpu_lab/results/desk/eager_scaled.json > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/c3762435fa004f0ba2eff0bef435f7d8.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/c3762435fa004f0ba2eff0bef435f7d8.exit
