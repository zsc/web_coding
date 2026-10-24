#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'cd /home/zsc/mnist_gpu_lab_20261024 && /home/zsc/miniconda3/envs/py312/bin/python src/train.py check --output results/correctness.json && /home/zsc/miniconda3/envs/py312/bin/python src/train.py bench --output results/benchmark.json' > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/645e4c0ababa4716995af5ca5c23cb87.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/645e4c0ababa4716995af5ca5c23cb87.exit
