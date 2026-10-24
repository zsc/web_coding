#!/bin/sh
scp -r desk:/home/zsc/ppo_gpu_lab/results ppo_gpu_lab/remote_results_latest > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/67a6913c067243b89581f96a67e1d81d.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/67a6913c067243b89581f96a67e1d81d.exit
