#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'find /home/zsc/miniconda3/envs /home/zsc -maxdepth 5 -path "*/site-packages/torch" -type d 2>/dev/null | head -20; ls -d /home/zsc/miniconda3/envs/*; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/a1ef7805fc0943f4a72a28f85cf2f9a0.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/a1ef7805fc0943f4a72a28f85cf2f9a0.exit
