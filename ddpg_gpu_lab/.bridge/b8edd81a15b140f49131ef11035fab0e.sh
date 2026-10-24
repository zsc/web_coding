#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && tail -6 results/bench.log' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/b8edd81a15b140f49131ef11035fab0e.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/b8edd81a15b140f49131ef11035fab0e.exit
