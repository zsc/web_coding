#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab && tar xzf implementation.tar.gz && /home/zsc/miniconda3/envs/py312/bin/python extract.py && /home/zsc/miniconda3/envs/py312/bin/python build_report.py' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/4e22acd299e9407d9c0bd057d99bf1bd.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/4e22acd299e9407d9c0bd057d99bf1bd.exit
