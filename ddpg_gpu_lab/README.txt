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

第一阶段原始记录（3 seeds；新的100-seed流程见文末）

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

2026-10-24 第二阶段：100 个独立训练 seed
  最新报告仍为 report.html；初始 3-seed 报告归档为 report_initial.html。
  用户允许采用 TD3 提高稳定性。最终配置以 run_100seeds.sh 为准。
  独立调参 seeds 1001–1016；正式训练固定 seeds 1–100，不替换失败种子。
  multiseed.py 将参数、优化器、replay 按 seed 维度隔离，随机流按 seed/step/stream 键控。
  TD3：双 critic + min target、actor/target 每两步更新、目标动作高斯平滑。
  另有显式输入/动作归一化、梯度范数上限10、镜像数据增强与奇对称 actor。
  pi(s) = 10*tanh([z(s)-z(-s)]/2)，z为激活前logit；网络仍是32/16。修复了先tanh再相减可能共同饱和、动作和梯度归零的问题。未使用 LQR、专家动作或模仿学习。
  环境物理、奖励与近直立 reset 分布不变；训练截断延长至2000步。
  原始 batch128，镜像后256；每seed32环境；actor/critic lr均3e-4。
  验证达标后冻结该seed全部策略/价值/目标权重，不继续更新导致遗忘。
  正式验证：64个初态，每条100000步；最终另用64个留出初态。
  压力测试：256个新初态 × 100000步；无噪声、无reset。
  原JS复核：四个边界初态 × 10000步，原样提取的环境类 + FP64 actor推理。
  以上均为有限时长、原reset范围内的测试，不是无限稳定证明。

  bash run_100seeds.sh
  python multiseed.py --help
  python verify_policies.py results/seeds100_td3.json --horizon 100000 --episodes 256 --split 997
  node verify_original.cjs results/seeds100_td3.json 10000 results/verification_original.json
  python build_robust_report.py

  正式训练结果中 actors 字段是100套最终保存权重；TD3推理必须应用 obs_scale 和 symmetry。
  完整续训checkpoint含优化器/replay，保存在desk对应结果目录。
  CPU线程1；TF32关闭。CUDA Graph捕获前后恢复所有状态，隐藏预热更新不计入训练。
  staged编译使replay原地scatter不被Inductor改写成两遍整池复制，实测约2.64倍改善。
  100-seed时间是整批墙钟时间，不是单个seed延迟；冻结者仍占计算槽位。
  单独列出训练主循环、初始化/编译、进程总耗时、长时评估和开发调试实验。
  完整训练进程计时包含Python/PyTorch导入，但不包含先运行的数值测试。
  共享GPU与编译缓存会影响速度，原始五轮数据和运行日志均保留。

失败尝试追踪
  results/seeds100_td3_posttanh_attempt01.json：先tanh再做奇对称，90k checkpoint通过93/100。
  seeds 3,10,33,59,65共同饱和：logit约-10，动作恒0，梯度消失；中止进一步空转。
  source/multiseed_posttanh.py保留该版本；最终版在logit层做对称化，并增加大共同偏置回归测试。
  results/td3_logit_pilot.json：修复后独立16 seed调参组，50k时验证与留出均撑满100k步。
  verify_policies.py与verify_original.cjs根据symmetry_mode区分两种导出策略，旧权重不会被静默误解释。

最终实测
  固定100 seeds全部通过64初态验证 + 64初态留出 + 256新初态压力集，每条100000步。
  原JS环境四个边界初态10000步复核：100/100。
  纯训练291.48797秒；setup54.67153秒；完整训练进程456.91秒。
  压力复测加载/编译11.12236秒，执行30.00427秒；原JS复核9.63902秒，均单独计量。
  达标停止更新分布：10k:14,20k:33,30k:31,40k:13,50k:5,60k:2,70k:1,80k:1。
  稳定口径为持续不越界；部分策略有偏移或振荡，不保证全部回到中心。
  同算法100模型学习更新交错微基准：72.23393 -> 3.35738 ms，21.515x。
  python benchmark_td3_paired.py 复现该五轮对照；不含环境/replay采样。
  ppo_neighbor_comparison.json记录邻目录PPO实测，非同任务同验收算法排名。
