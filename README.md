# Web Coding · GPU 实验档案

这是一个记录 GPU 与浏览器实验的静态站点仓库。每个实验都保留报告、核心源码和足以复核结论的原始记录，并在首页汇总入口。

**在线浏览：<https://zsc.github.io/web_coding/>**

> 这里的实验任务并不相同，指标不能跨项目直接比较。计时来自各报告记录的硬件与运行环境；共享 GPU、驱动、编译器和浏览器版本都可能改变绝对结果。

## 实验索引

| 实验 | 关注点 | 报告 |
| --- | --- | --- |
| MNIST GPU Lab | 小 CNN 的正确性、准确率提升、CUDA Graph 与训练吞吐 | [打开报告](https://zsc.github.io/web_coding/mnist_gpu_lab/report.html) |
| Diffusion GPU Lab | 扩散模型质量修正、DDIM 采样与 CUDA Graph 加速 | [质量报告](https://zsc.github.io/web_coding/diffusion_gpu_lab/report.html) · [速度报告](https://zsc.github.io/web_coding/diffusion_gpu_lab/report_speed.html) |
| MCMC Splatting GPU | 高斯 splatting 拟合、ROI MCMC、CUDA 与 WebGPU 实现 | [打开报告](https://zsc.github.io/web_coding/mcmc_splatting_gpu/report.html#webgpu-results) |
| DDPG GPU Lab | 倒立摆强化学习、TD3、多随机种子验收与批量更新 | [打开报告](https://zsc.github.io/web_coding/ddpg_gpu_lab/report.html) |
| PPO GPU Lab | 双倒立摆 PPO、CUDA 训练优化、WebGPU 训练与实时演示 | [完整报告](https://zsc.github.io/web_coding/ppo_gpu_lab/report.html) · [WebGPU](https://zsc.github.io/web_coding/ppo_gpu_lab/webgpu.html) · [实时实验](https://zsc.github.io/web_coding/ppo_gpu_lab/play.html) |
| CelebA Diffusion Lab | 64×64 人脸扩散、RGB/灰度分布诊断与负结果 | [打开报告](https://zsc.github.io/web_coding/celeba_diffusion_lab/report.html) |

## 目录结构

```text
.
├── index.html                 # GitHub Pages 首页
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
