#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd ~/ppo_gpu_lab && /home/zsc/.nvm/versions/node/v24.13.1/bin/node src/baseline.cjs > results/js_baseline.log && /home/zsc/miniconda3/envs/py312/bin/python src/validate.py && OMP_NUM_THREADS=1 /home/zsc/miniconda3/envs/py312/bin/python -u src/run.py --mode graph --envs 64 --horizon 32 --minibatch 64 --iterations 5 --out graph_matched' > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/5f0dc5da3ff1424b810f70af21aedb90.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/5f0dc5da3ff1424b810f70af21aedb90.exit
