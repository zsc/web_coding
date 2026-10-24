#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && for f in results/train_extended_seed*.log; do echo "$f"; tail -2 "$f"; done' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/f35c092dd4de4cbd8cfbf183b804a1bc.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/f35c092dd4de4cbd8cfbf183b804a1bc.exit
