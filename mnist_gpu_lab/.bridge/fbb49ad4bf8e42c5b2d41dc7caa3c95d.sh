#!/bin/sh
scp -r mnist_gpu_lab/data/mnist.npz desk:/home/zsc/mnist_gpu_lab_20261024/data/ > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/fbb49ad4bf8e42c5b2d41dc7caa3c95d.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/fbb49ad4bf8e42c5b2d41dc7caa3c95d.exit
