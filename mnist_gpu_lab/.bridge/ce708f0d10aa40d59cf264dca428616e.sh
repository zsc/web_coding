#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk '/home/zsc/miniconda3/envs/py312/bin/python -c "import torch,sys; print(sys.version); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name())"; lscpu | head -18; find /home/zsc -maxdepth 5 -name train-images-idx3-ubyte 2>/dev/null | head -10' > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/ce708f0d10aa40d59cf264dca428616e.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/mnist_gpu_lab/.bridge/ce708f0d10aa40d59cf264dca428616e.exit
