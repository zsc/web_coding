# Web Coding · GPU 实验档案

这是一个记录 GPU 与浏览器实验的静态站点仓库。当前收录 **6 组主实验**，共 **15 个报告与交互页面**；每个实验都保留报告、核心源码和足以复核结论的原始记录，并在首页汇总入口。

**在线浏览：<https://zsc.github.io/web_coding/>**

> 这里的实验任务并不相同，指标不能跨项目直接比较。计时来自各报告记录的硬件与运行环境；共享 GPU、驱动、编译器和浏览器版本都可能改变绝对结果。

## 运行环境

结果来自三类相互独立的运行环境，而不是在同一台机器上替换 API 的严格对照：

| 环境 | 实测配置 | 主要承担的实验 | 解读边界 |
| --- | --- | --- | --- |
| 网页 WebGPU（Apple M5） | Apple M5、10 个 GPU 核心、Chrome 155；网页 WebGPU/WGSL，MNIST 网页基线使用 WebGL / ANGLE Metal | PPO 的 WebGPU 训练与实时交互、MCMC 的 ROI + Adam WebGPU 页面、浏览器端吞吐与首载测量 | 计时可能包含浏览器、驱动、管线编译、回读和验证等页面开销；M5 与独立显卡是不同硬件，不能把 WebGPU/CUDA 的比值解释成 API 单项收益。 |
| 家用 GPU（desk） | NVIDIA RTX 2080 SUPER 8 GiB、AMD Ryzen 9 3900X、Ubuntu 22.04、PyTorch 2.10.0+cu128 | 主要 CUDA 实验：MNIST、Diffusion、MCMC、DDPG/TD3、PPO 与 CelebA 的训练、基准和验证 | 桌面同时运行图形程序或其他后台任务；绝对耗时会波动，不应视为独占硬件或理论性能上限。 |
| CPU（客户端或家用台式机） | 客户端 Apple M5 CPU，使用 Python 3.10 / PyTorch 2.8 CPU 做数值复核；家用台式机 Ryzen 9 3900X 承担 PPO CPU 基线、单线程 C++ MCMC 基线及部分数据与评估任务 | 正确性交叉检查、CPU 对照、报告生成、测试夹具和脚本编排 | CPU 基线通常没有做多核优化，线程数和精度也可能不同（例如 JavaScript Float64 对 PyTorch Float32）；不能仅凭一次 CPU/GPU 比值推断算法收益。 |

这些环境记录用于解释结果来源，不构成跨硬件排行榜。只有报告明确说明任务、配置、精度和计时范围一致时，跨环境数字才适合作定性参考。

## 实验索引

| 实验 | 关注点 | 报告 |
| --- | --- | --- |
| MNIST GPU Lab | 小 CNN 的正确性、准确率提升、CUDA Graph 与训练吞吐 | [打开报告](https://zsc.github.io/web_coding/mnist_gpu_lab/report.html) |
| Diffusion GPU Lab | 扩散模型质量修正、DDIM 采样与 CUDA Graph 加速 | [质量报告](https://zsc.github.io/web_coding/diffusion_gpu_lab/report.html) · [速度报告](https://zsc.github.io/web_coding/diffusion_gpu_lab/report_speed.html) |
| MCMC Splatting GPU | 高斯 splatting 拟合、ROI MCMC、CUDA 与 WebGPU 实现 | [打开报告](https://zsc.github.io/web_coding/mcmc_splatting_gpu/report.html#webgpu-results) |
| DDPG GPU Lab | 倒立摆强化学习、TD3、多随机种子验收与批量更新 | [打开报告](https://zsc.github.io/web_coding/ddpg_gpu_lab/report.html) |
| PPO GPU Lab | 双倒立摆 PPO、CUDA 训练优化、WebGPU 训练与实时演示 | [完整报告](https://zsc.github.io/web_coding/ppo_gpu_lab/report.html) · [WebGPU](https://zsc.github.io/web_coding/ppo_gpu_lab/webgpu.html) · [实时实验](https://zsc.github.io/web_coding/ppo_gpu_lab/play.html) |
| CelebA Diffusion Lab | RGB64 像素空间 U-Net / JiT；四组 16 卡训练各完成 512 万张次；质量、EMA、引导与训练加速 | [汇总报告](celeba_diffusion_lab/report.html) · [加速分析](celeba_diffusion_lab/profiling.html) |

## 专题页面

主索引已经列出的入口不再重复。下表只补充精度、交互工作台、原版基线和失败路径页面；它们不把不同基线或计时边界合并成一个排行榜。

| 页面 | 关注点 | 入口 |
| --- | --- | --- |
| MCMC WebGPU 精度实验室 | FP32、FP16 缓存与 FP16 exp 的轨迹、内存和全图误差，可在浏览器交互复测 | [精度实验室](https://zsc.github.io/web_coding/mcmc_splatting_gpu/webgpu_lab.html) |
| MCMC Adam ROI 工作台 | 6σ／4.5σ／4σ 裁剪与单／四 Worker 全图验收的现场对照 | [ROI 工作台](https://zsc.github.io/web_coding/mcmc_splatting_gpu/adam_roi_lab.html) |
| MNIST 浏览器原版 | 固定上游 TF.js 版本和资源地址，保留跨框架、跨硬件迁移的原始训练页 | [原版训练页](https://zsc.github.io/web_coding/mnist_gpu_lab/original_local.html) |
| PPO V1 | 从可运行 JS 基线到 CUDA Graph／融合 Adam 的首轮优化与数值检查 | [V1 报告](https://zsc.github.io/web_coding/ppo_gpu_lab/report_v1.html) |
| CelebA 灰度与分布归档 | 灰度、affine、颜色加权和早期 RGB 分布诊断；保留未解决问题与负结果 | [实验归档](https://zsc.github.io/web_coding/celeba_diffusion_lab/report_grayscale_archive.html) |

## 目录结构

```text
.
├── index.html                 # GitHub Pages 首页
├── assets/                    # 首页精选缩略图
├── mnist_gpu_lab/
├── diffusion_gpu_lab/
├── mcmc_splatting_gpu/
├── ddpg_gpu_lab/
├── ppo_gpu_lab/
└── celeba_diffusion_lab/
```

各实验目录通常包含以下内容：

- `report.html`：主要报告；部分实验另有速度、初始版本或交互页面。
- `src/`、`source/`：实验代码，以及固定来源的上游源码或版本快照。
- `results/`：JSON、日志、图片和图表等证据；部分文件只保留精选结果。
- `README.md`、`README.txt`：该实验的运行方式、依赖和结果边界。

## 本地预览

仓库首页和报告是静态文件。建议在仓库根目录启动本地服务器；
WebGPU、模块脚本和浏览器训练页面更适合通过 `localhost` 访问。

```sh
git clone https://github.com/zsc/web_coding.git
cd web_coding
python3 -m http.server 8000
```

然后打开 <http://localhost:8000/>。多数报告也可直接打开对应 HTML 文件离线阅读。

## 复现说明

- 先阅读目标实验目录内的 README 和报告，不同实验的 Python、PyTorch、CUDA、Node.js 与浏览器要求并不相同。
- GPU 实验需要与本机驱动兼容的 PyTorch/CUDA 环境；MCMC 的 WebGPU 和 PPO 交互页面需要支持 WebGPU 的现代浏览器。
- 历史 README 中的远端主机、环境名和绝对路径用于记录当时的实验环境，换机器运行时应替换为本机配置。
- 固定任务的结论应结合随机种子、数据划分、预热方式、同步点和计时范围一起查看。共享硬件上的绝对耗时不应视为通用性能上限。

## 数据与产物

仓库优先保存能够解释和复核结果的轻量文件：报告 HTML、源码、配置、日志摘要、JSON 指标和必要图表。

以下内容默认不提交：

- 完整 checkpoint、优化器状态、策略或权重包，以及导出的模型归档。
- 大型二进制文件、压缩包、重复的中间张量和可重新生成的数据集。
- QA 截图、响应式预览、临时脚本和未完成日志。

部分交互页面会内嵌演示所需的小型权重，以便直接运行；这不代表完整训练产物进入仓库。需要大文件时，优先在实验目录内生成，并通过 `.gitignore` 保持其本地状态。

## 发布

站点入口是仓库根目录的 `index.html`，通过 `main` 分支的 GitHub Pages 发布。更新首页或报告后推送到 `main` 即可触发站点更新：

```sh
git push origin main
```

新增实验时，请同时补充首页入口、可离线打开的报告、对应说明，以及经过筛选的证据文件。
