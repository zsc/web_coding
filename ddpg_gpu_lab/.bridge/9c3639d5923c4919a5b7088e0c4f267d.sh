#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && tail -8 results/bench.log 2>/dev/null; cat results/desk_baseline.json 2>/dev/null | head -28; pgrep -af "ddpg|baseline|validate"' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/9c3639d5923c4919a5b7088e0c4f267d.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/9c3639d5923c4919a5b7088e0c4f267d.exit
