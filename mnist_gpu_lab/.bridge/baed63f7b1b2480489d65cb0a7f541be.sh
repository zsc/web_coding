#!/bin/sh
scp -r desk:/home/zsc/mnist_gpu_lab_20261024/results/ mnist_gpu_lab/results/remote > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/baed63f7b1b2480489d65cb0a7f541be.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/baed63f7b1b2480489d65cb0a7f541be.exit
