#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && tail -4 results/bench.log' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/83b57de63e5e4c0b81dc95812ccbb5e9.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/83b57de63e5e4c0b81dc95812ccbb5e9.exit
