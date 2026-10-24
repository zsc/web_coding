#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && tail -18 results/validate.log; tail -10 results/bench.log 2>/dev/null' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/70b617657a0d4045b5d9fee98586cb2b.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/70b617657a0d4045b5d9fee98586cb2b.exit
