#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv' > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/0bf1f2fdc8954c0c9c4bbda59d94427c.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/0bf1f2fdc8954c0c9c4bbda59d94427c.exit
