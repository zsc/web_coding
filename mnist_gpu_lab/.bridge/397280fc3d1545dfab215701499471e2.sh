#!/bin/sh
scp -r desk:/home/zsc/mnist_gpu_lab_20261024/results/graph_10epochs mnist_gpu_lab/results/remote/ > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/397280fc3d1545dfab215701499471e2.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/397280fc3d1545dfab215701499471e2.exit
