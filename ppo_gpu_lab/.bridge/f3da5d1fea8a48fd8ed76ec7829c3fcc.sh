#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'nvidia-smi --query-gpu=timestamp,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv; lscpu | head -18; /home/zsc/miniconda3/envs/py312/bin/python -c "import torch; print(torch.__version__); print(torch.version.cuda)"' > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/f3da5d1fea8a48fd8ed76ec7829c3fcc.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/f3da5d1fea8a48fd8ed76ec7829c3fcc.exit
