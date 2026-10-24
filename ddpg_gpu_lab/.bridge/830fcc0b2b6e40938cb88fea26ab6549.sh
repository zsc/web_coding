#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && /home/zsc/miniconda3/envs/py312/bin/python debug_packed.py' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/830fcc0b2b6e40938cb88fea26ab6549.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/830fcc0b2b6e40938cb88fea26ab6549.exit
