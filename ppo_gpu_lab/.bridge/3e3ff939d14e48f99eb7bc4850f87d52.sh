#!/bin/sh
scp -r desk:/home/zsc/ppo_gpu_lab/results/validation_v2.json ppo_gpu_lab/results/validation_v2.json > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/3e3ff939d14e48f99eb7bc4850f87d52.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/3e3ff939d14e48f99eb7bc4850f87d52.exit
