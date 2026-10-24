#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && tail -4 results/train_extended_seed1.log; tail -2 results/train_extended_seed2.log 2>/dev/null' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/1e68c06cfff14a259444407c4cc0a8b6.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/1e68c06cfff14a259444407c4cc0a8b6.exit
