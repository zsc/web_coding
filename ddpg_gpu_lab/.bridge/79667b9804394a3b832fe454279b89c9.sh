#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && ls -lh results/validate_multiseed.log' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/79667b9804394a3b832fe454279b89c9.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/79667b9804394a3b832fe454279b89c9.exit
