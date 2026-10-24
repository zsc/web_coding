#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'mkdir -p /home/zsc/mnist_gpu_lab_20261024/src /home/zsc/mnist_gpu_lab_20261024/results /home/zsc/mnist_gpu_lab_20261024/data; /home/zsc/miniconda3/envs/py312/bin/python -c "import PIL,numpy; print(PIL.__version__,numpy.__version__)"' > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/7b694fdd39b1478c961322e67d409518.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/7b694fdd39b1478c961322e67d409518.exit
