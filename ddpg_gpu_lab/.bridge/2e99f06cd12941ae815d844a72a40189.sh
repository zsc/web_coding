#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && tail -2 results/train_extended_seed3.log' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/2e99f06cd12941ae815d844a72a40189.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/2e99f06cd12941ae815d844a72a40189.exit
