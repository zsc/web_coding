#!/bin/sh
scp -r desk:/home/zsc/mnist_gpu_lab_20261024/results/. mnist_gpu_lab/results/remote/ > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/381d8716babc47739dd2e0e4f38c27f0.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/381d8716babc47739dd2e0e4f38c27f0.exit
