#!/bin/sh
ssh -o BatchMode=yes -o ConnectTimeout=15 desk 'mkdir -p /home/zsc/ddpg_gpu_lab_20261024/results' > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/b87896120e054541bac58385ab63a73c.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ddpg_gpu_lab/.bridge/b87896120e054541bac58385ab63a73c.exit
