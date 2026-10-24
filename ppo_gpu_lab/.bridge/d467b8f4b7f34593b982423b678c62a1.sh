#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv; /usr/local/cuda/bin/nvcc --version; ls /usr/local; /home/zsc/miniconda3/envs/py312/bin/python -c "from torch.utils.cpp_extension import CUDA_HOME; import torch; print(CUDA_HOME,torch.version.cuda)"' > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/d467b8f4b7f34593b982423b678c62a1.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/d467b8f4b7f34593b982423b678c62a1.exit
