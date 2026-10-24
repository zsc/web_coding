#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && tail -7 results/seeds100_ddpg_long.log; pgrep -af "multiseed.py"' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/f5ea2337da9e4260beb61d65cf22ec09.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/f5ea2337da9e4260beb61d65cf22ec09.exit
