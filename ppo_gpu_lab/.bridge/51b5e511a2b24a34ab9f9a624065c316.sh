#!/bin/sh
scp -r desk:/home/zsc/ppo_gpu_lab/results/benchmark_v2.json ppo_gpu_lab/results/benchmark_v2.json > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/51b5e511a2b24a34ab9f9a624065c316.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/51b5e511a2b24a34ab9f9a624065c316.exit
