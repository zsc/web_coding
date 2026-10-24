DDPG GPU Lab — 2026-10-24

打开 report.html 即可查看完整中文报告（离线，不依赖 CDN）。

本地目录：/Users/zsc/Downloads/web_coding/ddpg_gpu_lab
远端目录：/home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab

源文件：zsc/zsc.github.io/widgets/ddpg_invpendulum.html
Git blob：6409582ccaf159a822a422bec7f8fe2ed9fda144
source/original.html 为完整快照，extract.py 原样提取计算类到 core_original.cjs。

已验证环境
  desk: RTX 2080 SUPER / 8 GiB, Ryzen 9 3900X, Ubuntu 22.04
  Python: /home/zsc/miniconda3/envs/py312/bin/python
  PyTorch: 2.10.0+cu128, FP32, CPU线程1
  Node: /home/zsc/.nvm/versions/node/v24.13.1/bin/node (v24.13.1)
  本机数值核对额外通过 Python3.10 / PyTorch2.8 CPU。

运行
  cd /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab
  bash run_desk.sh

各阶段
  python extract.py
  node baseline.cjs --mode=fixture --out=results/fixture.json
  python validate.py --cuda
  node baseline.cjs --mode=bench --out=results/desk_baseline.json
  python ddpg.py bench --updates 300 --out results/desk_benchmark.json
  python bench_full.py
  python ddpg.py train --device cuda --compile-full --graph --envs 32 --batch 256 --updates 20000 --seed 1 --out results/train_gpu_seed1.json
  node baseline.cjs --mode=train --steps=20000 --seed=1 --out=results/train_original_seed1.json
  python build_report.py

本机 SSH 网络上下文
  当前 Codex 子进程直接 ssh desk 返回 No route to host。
  用户已有 tmux 连接有效；bridge.py 通过同一 tmux 服务新建临时窗口执行 SSH/SCP。
  不改 SSH 配置，不关闭原来的连接。需要 tmux 服务已经运行。
  python3 bridge.py run 'nvidia-smi'
  python3 bridge.py push ddpg.py /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab/ddpg.py
  python3 bridge.py pull /home/zsc/ddpg_gpu_lab_20261024/ddpg_gpu_lab/results/ .

算法边界
  原 JS: batch64 内 critic 连续做64次SGD；actor一次平均梯度上升。
  张量版: critic一次minibatch平均梯度，actor一次；默认Adam。
  --optimizer sgd 是 minibatch SGD，不等价于原版串行SGD。
  validate.py 中 serial_update 实现原版串行规则，用于精确对照。
  维持小网络32/16、原奖励/物理/均匀噪声/非终止时间截断。
  默认单环境200步截断；评估固定64个确定性起点，无探索噪声。
  原环境常量目前固定；未移植网页所有UI可修改超参。
  原版非有限actor梯度跳过改为每个日志间隔检测并报错。
  GPU32env版更新/新样本比例1/32，原版1/1，样本复用率不同。
  数据与模型保持GPU驻留，无逐步 .item() 或 CPU 回传；只在评估/日志同步。
  replay容量50000，11个FP32字段，2.10MiB；CUDA运行时和图额外计入显存。
  CUDA Graph要求固定环境数、batch和地址；改shape应重新编译/捕获。
  --compile 编译学习更新；--compile-full 编译整个采集+学习循环。
  保留随机探索与回放采样；不是在重复固定样本伪造训练吞吐量。

证据与限制
  results/*.json 是机器可读原始数据，*.log 保留实际运行输出。
  编译GPU与eagerGPU的同配置比较可以评估系统优化。
  原JS vs 张量版同时改变训练规则；不宣称严格等价倍数。
  编译和预热成本独立计量；短任务必须看包含setup的总耗时。
  GPU训练结果有3个种子，不挑最好中间checkpoint替换最终权重。
  200步成功不等于无限稳定；另报告1000步评估。
  共享桌面GPU含图形程序和其他任务；五轮结果有波动，不是独占极限。
  不安装全局依赖、不更改系统配置、不终止任何既有GPU任务。

延长预算实验
  三个种子独立从头运行100000次更新，参数不变。
  python ddpg.py train --device cuda --compile-full --graph --envs 32 --batch 256 --updates 100000 --eval-every 10000 --seed 1 --out results/train_extended_seed1.json
  另对seed2、seed3重复；没有覆盖20k预算的早期失败结果。
  setup使用现有编译缓存，并非严格冷启动。

直接使用保存的策略
  python evaluate_saved.py results/train_extended_seed1.json --device cpu --horizon 1000 --out results/saved_policy_eval.json
  文件内actor字段包含6个数组 W1,b1,W2,b2,W3,b3。
  保存的是最终actor权重与指标，不是用于无缝续训的完整优化器/replay checkpoint。
