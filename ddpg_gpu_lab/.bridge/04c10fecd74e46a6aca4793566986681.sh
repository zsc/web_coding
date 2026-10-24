#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && /home/zsc/miniconda3/envs/py312/bin/python validate_multiseed.py --cuda > results/validate_multiseed.log 2>&1 && /home/zsc/miniconda3/envs/py312/bin/python multiseed.py --seeds 100 --updates 100000 --eval-every 25000 --out results/seeds100_ddpg > results/seeds100_ddpg.log 2>&1' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/04c10fecd74e46a6aca4793566986681.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/04c10fecd74e46a6aca4793566986681.exit
