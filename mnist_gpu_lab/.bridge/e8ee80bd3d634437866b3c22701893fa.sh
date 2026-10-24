#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/mnist_gpu_lab_20261024 && /home/zsc/miniconda3/envs/py312/bin/python src/train.py train --mode cuda_graph --epochs 10 --output results/graph_10epochs && nvidia-smi > results/gpu_after.txt' > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/e8ee80bd3d634437866b3c22701893fa.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/e8ee80bd3d634437866b3c22701893fa.exit
