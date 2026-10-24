#!/bin/sh
scp -r desk:/home/zsc/mnist_gpu_lab_20261024/results/ mnist_gpu_lab/results/remote/ > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/5f123fae7fa9454b85e5d3d32a11bafd.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/5f123fae7fa9454b85e5d3d32a11bafd.exit
