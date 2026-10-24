#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv' > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/a98222705fc34388b1e5091ebac4e292.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/a98222705fc34388b1e5091ebac4e292.exit
