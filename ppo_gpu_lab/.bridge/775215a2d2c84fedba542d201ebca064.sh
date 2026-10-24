#!/bin/sh
scp -r ppo_gpu_lab/src/diagnose.py desk:/home/zsc/ppo_gpu_lab/src/ > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/775215a2d2c84fedba542d201ebca064.log 2>&1
echo $? > /Users/zsc/Downloads/web_coding/ppo_gpu_lab/.bridge/775215a2d2c84fedba542d201ebca064.exit
