# MNIST GPU Lab

本地入口：`report.html`，可直接双击，图表和测试图片均内嵌。需要原网页训练 UI 时，在本目录运行 `python3 -m http.server 8767 --bind 127.0.0.1`，访问 `http://127.0.0.1:8767/original_local.html`。

## 准确率升级（追加实验）

`src/accuracy_train.py` 保留 CUDA Graph，加上可原位更新的 GPU 学习率 tensor、余弦衰减和逐图随机仿射增强。先从训练集分出每类 500 张、共 5,000 张验证集，以剩余 55,000 张比较四个预先指定的方案；全部训练 20 轮，按最后一轮验证准确率选配置，平局时按验证交叉熵。选定后才加载本轮测试数据，使用全部 60,000 张分别以预先指定的 seed 42、43 重训。

| 20 轮方案 | 验证集准确率 |
| --- | ---: |
| 原宽度，固定 lr=0.001 | 98.78% |
| 原宽度，余弦 lr 衰减 | 99.22% |
| 16/32/64 + Dense128，余弦衰减 | 99.36% |
| 加宽 + 余弦衰减 + 轻量仿射增强 | 99.56% |

选中最后一个方案。lr 从 0.001 衰减至 0.00005；训练增强为 ±10° 旋转、约 ±2px 平移、0.9–1.1 缩放，不翻转。验证与测试不增强。新模型参数 98,442，原模型 25,034。

完整测试集结果：seed 42 **99.50%（50 个错误）**，seed 43 **99.54%（46 个错误）**；训练分别约 20.46 / 20.70 秒。首轮交付模型为 98.36%（164 个错误）。两模型等权集成 99.53%，没有超过 seed 43，因此推荐单模型；默认交付预先指定的 seed 42，没有按测试分数挑选 seed。

```sh
cd /home/zsc/mnist_gpu_lab_20261024
/home/zsc/miniconda3/envs/py312/bin/python src/accuracy_train.py search --epochs 20
/home/zsc/miniconda3/envs/py312/bin/python src/accuracy_train.py final
```

结果在 `results/accuracy/`，新模型在 `results/accuracy/final_seed42/model.pt`；兼容浏览器的导出在 `exports/tfjs_accuracy/`。原有模型未覆盖。验证了 Graph 中学习率改变、lr=0 不更新参数、增强随机数在不同回放间变化、小尾 batch 后继续回放、全部测试概率的 checkpoint 复读一致性及 TF.js 权重转换一致性。

本轮选型不用测试集，但本项目此前已展示过测试结果，不能声称它在整个项目历史中从未被查看。准确率收益与训练成本都按实际结果报告；GPU 仍为共享负载。

## 文件

- `original/train.html`：固定上游版本，`provenance.json` 记录 commit 和 SHA256。
- `original_local.html`：仅替换资源地址并固定 TF.js 版本的原版 demo；labels 仍从外部 JS 加载，PNG 走 HTTP。
- `src/data.py`：IDX → 无缝拼图；或下载固定版本拼图，校验 Git blob hash，恢复 uint8 tiles，生成 NPZ 缓存。
- `src/train.py`：可配置的 1–3 层 CNN；CPU、CUDA eager、fused Adam、CUDA Graph、channels_last、AMP。
- `src/tfjs_parity.cjs`：将 PyTorch 权重转回原 TF.js 网络，验证概率并生成原生 TF.js 导出。
- `results/remote/`：实测 JSON、checkpoint、全测试集概率；`exploration/` 是前期 GPU shuffle 实验，不混入最终 CPU-shuffle 主对照。
- `exports/tfjs/`：原生 TF.js model.json + weights.bin。
- `src/build_report.py` / `src/report_template.html`：生成离线报告。
- `bridge.py`：通过现有 tmux server 的网络上下文调用 ssh/scp，任务使用独立临时窗口，不占用用户已连接的 pane。

## desk 运行

已部署在 `/home/zsc/mnist_gpu_lab_20261024`。使用已有环境 `/home/zsc/miniconda3/envs/py312/bin/python`，没有改全局 Python、驱动或用户 GPU 进程。

```sh
cd /home/zsc/mnist_gpu_lab_20261024
/home/zsc/miniconda3/envs/py312/bin/python src/train.py train \
  --mode cuda_graph --filters 8,16,32 --dense 64 --kernel 3 --pool 2 \
  --lr 0.001 --batch 128 --epochs 5 --output results/custom

/home/zsc/miniconda3/envs/py312/bin/python src/train.py check --output results/correctness.json
/home/zsc/miniconda3/envs/py312/bin/python src/train.py bench --output results/benchmark.json
```

输出目录会写入/覆盖同名文件；重复实验请使用新目录。全部实验使用 seed 42，CPU 生成每轮排列后一次传入 GPU，完整数据尾 batch 为 96。图预热、捕获后原位恢复参数与 Adam 状态；持久梯度地址保证尾 batch 后回放正确。

## 数据

```sh
python src/data.py --data data
# 如果已有未压缩 IDX（包含图像和标签）：
python src/data.py --idx-dir /path/to/idx --data data
```

本次在检查过的本地/远程数据位置没有找到 IDX，使用仓库同一 commit 下已有的四份 PNG/JS 数据，并验证 Git blob hash 和 SHA256。训练图 6860×6860、每行 245 张，尾部 25 个空 tile 不作为样本。测试图 2800×2800、每行 100 张。每张 28×28，无间隔、caption 或缩放。

## 依赖与环境

desk 实测：Python 3.12.12，PyTorch 2.10.0+cu128，Pillow 12.1.1，NumPy 2.4.2；RTX 2080 SUPER 8 GiB，Ryzen 9 3900X，CPU 4 线程。报告生成只需 NumPy；数据处理另需 Pillow。浏览器基准脚本使用本机 Chrome 和 Codex 已有的 Playwright 路径，可按本机安装位置修改。

## 测量口径

完整训练对照：60,000 / 10,000，5 epochs，batch 128，8/16/32 通道，Dense 64，Adam lr 0.001。普通 CUDA 和 FP32 Graph 各跑三次；CPU 和 AMP 各一次。另有 10 epochs 补充实验和三次网页默认 6,000 / 1,000 子集训练。

浏览器基线真实运行原版 trainModel，TF.js 4.22.0，Chrome 155，硬件 WebGL / ANGLE Metal Apple M5；第一次包含 shader 编译，第二、三次是热运行。GPU 输入已经预处理为 NPZ，PNG 预处理不计入训练时间。跨浏览器/desk 的比值是硬件和框架整体迁移收益，不能说成 Graph 单项收益。

完整 GPU 所有计时均同步 CUDA。报告分别展示训练+测试及含加载/建模/图捕获的耗时，不含进程启动、import、下载、首次 PNG 解码转换及最终导出/复读检查。训练 accuracy 是在线 batch 准确率，测试 accuracy 是轮末完整评估。未开启确定性卷积，因此重复运行准确率略有差异。

GPU 同时运行其他进程；没有停掉游戏、桌面或其他任务。保留所有不利结果和波动范围。FP16 使用静态 scale 128 并检查每轮有限值；不是针对任意网络/学习率的动态 GradScaler。

## 已通过的正确性检查

- train/test tile 完整往返一致；首尾及跨行位置与直接 crop 一致。
- CPU / GPU logits 最大绝对误差约 1.19e-7。
- FP32 fused eager / Graph 五步测试，包含尾 batch 后继续回放；权重最大误差约 4.99e-7，Adam 状态约 1.68e-8。
- 交付 checkpoint 重新加载后，全部 10,000 张测试概率完全一致。
- 14 张跨 tile 行界/末尾的图像转回 TF.js CPU 网络，概率最大误差约 1.19e-7，分类完全一致。

PyTorch 与 TF.js 保持模型结构/目标等价，但 RNG、Adam epsilon 放置与内核浮点运算不保证逐步一致；跨框架检查使用相同导出权重。
