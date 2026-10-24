#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && /home/zsc/.nvm/versions/node/v24.13.1/bin/node baseline.cjs --mode=train --steps=20000 --seed=1 --out=results/train_original_seed1.json > results/train_original_seed1.log 2>&1' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/82c6b067b4bc48b0b1bae90d7c82887c.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/82c6b067b4bc48b0b1bae90d7c82887c.exit
