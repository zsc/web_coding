#!/bin/sh
scp -r desk:/home/zsc/ppo_gpu_lab/results ppo_gpu_lab/remote_results > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/a92de4053d2440a49cbc5095584cb99e.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/a92de4053d2440a49cbc5095584cb99e.exit
