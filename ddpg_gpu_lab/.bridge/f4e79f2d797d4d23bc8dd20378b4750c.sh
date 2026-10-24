#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && tail -3 results/train_original_seed1.log; tail -3 results/train_extended_seed1.log' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/f4e79f2d797d4d23bc8dd20378b4750c.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/f4e79f2d797d4d23bc8dd20378b4750c.exit
