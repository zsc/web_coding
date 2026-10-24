#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'for p in /home/zsc/miniconda3/envs/*/bin/python; do "$p" -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>/dev/null && echo "$p"; done; lscpu | head -20' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/2eb2f822989d47f9b66ffc0471dc067e.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/2eb2f822989d47f9b66ffc0471dc067e.exit
