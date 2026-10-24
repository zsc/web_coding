# MCMC splatting: extraction, fast Chinese fitting, and WebGPU precision lab

Open `report.html` for the final report and the **live WebGPU ROI + Adam fitter**. The default target is the exact 320×128 “周舒畅” bitmap used on desk. Default settings are 1,600 Gaussians, 8 chains, 10,000 ROI MCMC steps and 8,000 Adam steps. Start from seeded target-weighted random initialization; no prefit parameters are loaded.

Serve this directory with `python3 -m http.server 8766 --bind 127.0.0.1` and open `http://127.0.0.1:8766/report.html#playground`. The target, WGSL and UI code are embedded in the report. WebGPU availability still depends on the browser and secure-context support.

## Desk local-convexity experiment: block Gauss–Newton / LM

Implemented and measured on RTX 2080 SUPER in `src/lm_fit.cu`. It reuses the existing ROI renderer and Adam kernel from `src/roi_fit.cu` (the only change to that file is a main-function include guard). Each Gaussian accumulates a 6-vector gradient and the 21 unique entries of `J_j^T J_j`; an equilibrated FP32 Cholesky solve computes a diagonally damped step. Steps are capped, projected to parameter bounds and evaluated by a fresh whole-chain render. Each chain accepts only a decrease in the continuous 6-sigma-rendered loss, otherwise increases damping. **Cross-Gaussian normal-matrix blocks are omitted:** this is a block approximation with LM-style damping, not an exact coupled global LM solve or a globally convex fitting problem.

Same target, 1,600 Gaussians, eight chains, three rotated repetitions, seed base 42:

| Method | Adam + LM steps | Median full process seconds | Median original MSE |
| --- | ---: | ---: | ---: |
| Pure Adam | 8,000 + 0 | 3.746 | 1.3280 |
| Block GN / LM | 0 + 3,456 | 3.712 | 5.1543 |
| Short Adam → block LM | 1,000 + 3,120 | 3.813 | 3.4140 |

The measured budgets differ from the baseline by −0.89% / +1.79%. Step counts were calibrated from pilot **total process times**, then frozen. The original experiment manifest's description says rounding to 32; the actual fixed counts follow rounding to 16. The summary records this documentation correction without altering the raw timings or settings, and the reproduction script now uses the corrected description. Three additional paired seed ensembles (1042, 2042, 3042, each with eight chains) all favored pure Adam too. A longer pilot with 6,000 Adam + 900 LM also lost at a similar budget (3.78 s, MSE 1.4883). All exploratory trials are retained, including unfavorable results.

The practical result is to keep pure Adam for this target. The block approximation did not reduce iterations enough to pay for its additional curvature and trial-evaluation work. Omitting overlap couplings is a plausible limitation, not a proven causal ablation. Coupled/matrix-free global Gauss–Newton and a separate amplitude-only convex solve were **not tested** here.

Timing uses `perf_counter` in a parent Python process around the complete executable: launch, CUDA context, initialization, fitting, readback, full FP64 validation, output files, cleanup and exit are included. Compilation, SSH and harness setup are excluded. Existing graphics processes were left alone; GPU utilization before/after was 38% / 41%. These timings should not be compared with historical runs as if hardware load were controlled.

Validation:

- `node src/check_lm.mjs`: 32 actual GPU local systems checked against independent Float64 full-image central differences. Maximum relative L2 errors: gradient 6.99e−7, Gauss–Newton matrix 1.76e−7; maximum absolute update difference 1.48e−5.
- `node src/check_lm_outputs.mjs`: all 19 complete outputs (one warmup, nine timing repeats, nine holdout runs) match the verbatim upstream renderer/MSE, with zero differing pixels. CUDA checks every accepted LM step for monotonic continuous loss; the independent script checks exported curves and acceptance counts.
- `results/desk_lm_study/summary.json` contains the complete summary, validation and raw metadata. Full outputs and images are in that directory; exploratory runs and the linearization audit are in `results/desk_lm/`. `source/desk_lm_study/` preserves the exact measured sources, verified against the manifest hashes.

```sh
./run_lm_study.sh
# If this terminal cannot reach desk, use the existing tmux environment:
tmux run-shell -b '/Users/zsc/Downloads/web_coding/mcmc_splatting_gpu/run_lm_study.sh > /Users/zsc/Downloads/web_coding/mcmc_splatting_gpu/results/desk_lm_study.log 2>&1'
# On desk after compilation:
./lm_fit --gaussians 1600 --chains 8 --adam-steps 0 --lm-steps 3456 --output results/lm_replay
```

The report includes the comparison, held-out seeds, fitted images/error maps and all pilot results at `#desk-lm-results`. WebGPU's existing implementation remains unchanged by this CUDA experiment.

## Browser ROI + Adam acceptance

### Optimizer ablation: pure Adam versus MCMC → Adam

The same M5 browser, target, 1,600 Gaussians, eight initial chains and FP32 were compared in three rotated rounds. All timed trials reuse warmed pipelines and include initialization, GPU work, readback and full Float64 validation.

| Method | MCMC + Adam steps | Median total seconds | Full MSE |
| --- | ---: | ---: | ---: |
| MCMC → Adam | 10,000 + 8,000 | 3.806 | 1.334888 |
| Pure Adam, equal Adam steps | 0 + 8,000 | 3.688 | 1.335669 |
| Pure Adam, matched total time | 0 + 8,288 | 3.791 | 1.309985 |

At equal Adam steps, removing MCMC reduces median time by 3.10%, with only 0.000781 higher MSE. At matched total time (−0.38% measured difference), pure Adam reduces MSE by 1.87%. For this target and target-weighted initialization, MCMC warm-start is not necessary. This is a fixed-target, fixed-eight-seed experiment; the three repetitions measure timing variability, not independent initializations. It does not establish that MCMC never helps on other targets, initialization policies or shorter budgets. Desk was not rerun for this ablation.

Use the report's **消融：纯 Adam / 同步数 / 同耗时** button to reproduce. It warms the pipeline, calibrates using the measured total time of the hybrid baseline and pure Adam at 8,000 / 10,000 steps, freezes a time-only interpolated step count (rounded to 32), then executes three rotated rounds. It never uses MSE to choose the matched step count, and never sleeps to pad the timer. Every trial starts over; the cosine learning-rate policy is normalized to that trial's total Adam steps, so 8,288 steps is a longer fresh schedule, not a continuation after 8,000. Other devices should recalibrate rather than assume 8,288 matches their time.

`results/webgpu_ablation.json` retains warmup, calibration, all nine timed runs and a complete final parameter/image representative per method. `results/webgpu_ablation_summary.json` includes medians, ranges and source hashes; `source/webgpu_ablation/` archives the measured sources. `node src/check_ablation.mjs` reproduces the independent upstream-renderer check: all three representatives have exactly the same MSE and zero differing pixels. Two early pilot runs are retained as `results/webgpu_ablation_pilot_*.json` and are not included in the reported nine-run statistics. The historical hybrid default remains available for reproducibility; set MCMC=0 and Adam=8,288 to run the matched-time pure-Adam case directly.

### Backend and original acceptance measurements

The live backend is `src/hybrid_webgpu.mjs`, with UI in `src/hybrid_playground.mjs` / `.html`. It ports the desk optimizer to 32-thread per-Gaussian workgroups and subgroup reductions (shared-memory fallback when unavailable). ROI MCMC runs 256 proposals per dispatch; Adam batches 128 iterations per submission, with progress readback about every 250 ms. Rendering uses u32 atomic accumulation in 1/1024 intensity units. With amplitude ≤255 and N≤10,000, the per-pixel accumulator cannot exceed 2,611,200,000. Optimizer parameters and moments remain FP32.

The default is FP32. Optional FP16 exp/amplitude and FP16 ROI exponent-cache paths are implemented, followed by 1,000 FP32 Adam polishing steps. Both were measured; on this Apple M5, half precision does not give a stable speedup, and the extra ROI-cache traffic is slower. The much larger improvement comes from sparse ROI work and parallel Adam. See `results/webgpu_hybrid_summary.json` for measured first-load and warm medians, memory, MSE and source hashes; `results/webgpu_hybrid_runs.json` retains both the initial nine runs and a fully warm nine-run comparison.

Final parameters are independently rerendered over the **entire image with JS Float64, without truncation**, in a Worker. Validation uses the original round/clip MSE. Pipeline creation, setup, GPU work, readback and this check are included in first-load timing; page loading and export-file serialization are excluded. Each first-load repeat reloads the page and creates a new device/pipeline; browser/driver compilation caches may remain. GPU time uses timestamp queries. The selected chain is chosen by quantized GPU loss before full validation, as on desk. Fixed-point rendering and floating-point arithmetic mean browser/CUDA trajectories are not identical.

`results/webgpu_hybrid_cold_{1,2,3}.json` retains complete first-load outputs: 3.941 / 3.966 / 4.005 seconds, each at MSE 1.3348876953125. `results/webgpu_hybrid_upstream_check.json` also rerenders the browser parameters using the verbatim upstream renderer/loss, with identical MSE and zero differing pixels. `results/webgpu_validator_check.json` compares the independent validator with desk's stored parameters and exactly matches its MSE. `node src/check_hybrid_gradient.mjs` compares an actual GPU single Adam step with independent full-image central finite differences: 191 nondegenerate parameters pass; one near-zero derivative is explicitly excluded because the normalized Adam step is ill-conditioned at numerical zero. `results/webgpu_hybrid_stop_check.json` records stopping a 10,000-Gaussian / 10-million-step request.

Controls retain the raised 10,000-Gaussian and 10-million-step limits (for both MCMC and Adam). Those limits are supported inputs, not claims of completing every allowed configuration in 10 seconds. The full-image MCMC precision laboratory in `webgpu_lab.html` remains available as a historical experiment.

## Final acceptance: 周舒畅 in under 10 seconds

The fixed 320×128 target, 1,600 Gaussians, 8 chains, 10,000 local MCMC steps per chain plus 8,000 parallel Adam steps passes three separate process runs:

| Run | Full process seconds (`/usr/bin/time`) | Full FP64 rerender MSE |
| --- | ---: | ---: |
| 1 | 7.62 | 1.34331 |
| 2 | 8.13 | 1.33965 |
| 3 | 8.19 | 1.35791 |

This improves the earlier 77.26-second GPU-only MCMC result (MSE 95.89). Compilation, transfer, font rasterization and report construction are excluded; initialization, fitting, readback, validation, output and process teardown are included in the final process timer. Concurrent GPU graphics load remained present.

```sh
./run_fast.sh
# On desk, after compilation:
./hybrid_fit --input zhou_target.pgm --gaussians 1600 --chains 8 \
  --steps 10000 --adam-steps 8000 --output results/fast
```

`src/roi_fit.cu` implements 6-sigma local support, delta-SSE updates, warp reductions, sparse splatting and analytic Adam gradients. It includes the shared definitions from `desk_benchmark.cu`. Adam optimizes a smooth, unquantized/unclamped intensity objective; this is an optimizer change, not an equivalent MCMC trajectory. Final exported pixels and MSE are independently computed from the chosen parameters with the original full, untruncated FP64 Gaussian renderer and round/clip objective. The selected chain is chosen using GPU cached loss before full validation. `results/gradient_check.json` records a finite-difference check of all six derivatives.

`run_acceptance.sh` reproduces three final runs and samples the renderer/gradient kernels with Nsight Compute. `results/final_summary.json` includes raw run metadata, final source hashes and browser precision measurements. `assets/zhou_final.png` is the representative median-time output; `assets/zhou_error_x16.png` shows absolute error magnified 16×.

The final sampled render+gradient pair has about 14.97% executed FP32 arithmetic utilization, with SM throughput 73.23% / 66.33% and occupancy 83.97% / 83.07%. These are kernel samples, not model MFU or whole-process averages. The earlier 3.85% number below describes the first full-image MCMC kernel only.

## Earlier full-image WebGPU / FP16 experiment

`src/webgpu.mjs` and self-contained `webgpu_lab.html` implement real device-resident MCMC in WGSL. On the local Apple M5 adapter, FP32 vs FP16-cache vs FP16-exp GPU medians were 116.33 / 101.84 / 101.97 ms for 160×64, 128 Gaussians, 4 chains, 8,192 steps; each has 3 interleaved repetitions after warmup. FP16 cache halves contribution memory (20 → 10 MiB), increasing throughput ~14.2%; FP16 exp adds no clear benefit. All modes use FP32 parameters/sums and integer SSE. Different arithmetic produces different trajectories; final loss and image error are recorded in `results/webgpu_precision.json`.

WGSL has optional f16 but no standard native bf16 scalar. BF16 emulation is not claimed as hardware acceleration. The CUDA ROI + Adam pipeline is now ported separately in `hybrid_webgpu.mjs`. The earlier MCMC-only measurements in this section are distinct from the new report backend; CUDA timings are not presented as browser timings.

```sh
python3 src/build_webgpu_lab.py
python3 -m http.server 8766 --bind 127.0.0.1
# Open http://127.0.0.1:8766/webgpu_lab.html
```

Upstream HTML is pinned in `source/provenance.json`. It fits a 2D grayscale image with additive rotated Gaussians, quantizes/clamps the total intensity, and uses single-parameter random proposals with a Metropolis-style acceptance rule. Clamped boundary proposals lack a Hastings correction; this is evaluated as a stochastic optimizer, not a guaranteed exact sampler.

## Run locally

```sh
node src/benchmark.mjs
node src/quality.mjs
python3 src/summarize_final.py
python3 src/finalize_report.py
open report.html
```

Node 26 and system Python 3, no npm or pip dependencies. `core.mjs` exports `CachedChain`, `initialize`, `propose`, and deterministic fixture generation. The cache stores unquantized sums and per-Gaussian contributions, preserving final quantization. It reduces each proposal from O(PN) to O(P), with O(PN) storage. Floating-point accumulation can still affect rounding thresholds; equality claims are limited to tested cases.

## Earlier full-image MCMC benchmark on desk

```sh
./run_desk.sh
./run_fit.sh
```

Scripts copy only this project's files to `~/mcmc_splatting_gpu_20261024`. Build uses `/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false`. Desk has RTX 2080 SUPER 8 GiB, CUDA 12.8 and Ryzen 9 3900X. Default dtype is FP64; `--fp32` explicitly selects approximate arithmetic.

If direct SSH from Codex returns `No route to host` while the existing tmux environment can connect, this session successfully used:

```sh
tmux run-shell -b '/Users/zsc/Downloads/web_coding/mcmc_splatting_gpu/run_desk.sh > /Users/zsc/Downloads/web_coding/mcmc_splatting_gpu/results/desk_run.log 2>&1'
```

The existing interactive SSH pane was left intact. No driver settings or other GPU processes were changed.

On desk, after compilation:

```sh
cd ~/mcmc_splatting_gpu_20261024
./desk_benchmark --chains 32
./desk_benchmark --chains 1 --fp32
./desk_benchmark --fit --fp32 --chains 32 --gaussians 100 --steps 100000 --output results/fit
./desk_benchmark --fit --fp32 --input target.pgm --chains 32 --gaussians 100 --steps 100000 --final-temperature 0.05 --output results/custom
```

Input is 8-bit binary PGM (P5, maxval 255); export one from the report. Input dimensions override CLI dimensions for fitting. Output directories must exist. Outputs are a grayscale PGM and JSON parameters in `[x,y,scaleX,scaleY,rotation,opacity]` order. Each final chain is independently rerendered in FP64, and the lowest final MSE is exported. The fitter selects final states, not the best state over every iteration.

One CUDA block owns one chain; threads evaluate pixels, reduce integer SSE, and commit after a uniform accept decision. Up to 256 iterations execute per launch. No simultaneous acceptance of overlapping proposals occurs. Memory is bounded to half of currently free VRAM. This first kernel targets small images/multiple chains; high-resolution performance is not validated.

## Evidence and limits

- `results/local_benchmark.json`: 4 x 500-step edge-case checks and 3 interleaved timings at each of 3 scales, same initial states/proposal streams. Renderer and loss are extracted verbatim from upstream and run in the same JS realm.
- `results/desk/*_fp*.json`: C++ CPU vs CUDA, same dtype and chain count, 3 repetitions, 500-step full-render oracle, final all-chain loss/acceptance comparison. FP64 checks pass. FP32 has loss/trajectory differences and is labeled approximate.
- `results/quality.json`: fixed vs annealed 25-Gaussian runs with same 10k budget, plus a 100-Gaussian 100k run. Seed-specific observations, not multi-seed claims.
- `results/desk/fit.{json,pgm}`: 32 x 100k-step annealed GPU fit. 2.807 s kernel time, final selected MSE 610.204, maximum final cached-vs-rerendered MSE difference 0.0278.
- `results/desk/ncu_privileged.csv` and `results/mfu.json`: one representative FP32 kernel sampled by Nsight Compute. Ordinary access was denied; existing passwordless sudo enabled scoped counter access without changing the driver policy. `profile_desk.sh` reproduces collection.

FP32 arithmetic utilization is `(FADD + FMUL + 2*FFMA) / kernel_seconds / (48*64*2*measured_SM_clock) = 3.85%`. This is executed arithmetic utilization (includes library arithmetic), **not model MFU**; it excludes integer, SFU and FP64 operations. SM throughput 9.48%, achieved occupancy 24.98%. Profile conditions and concurrent GPU load affect metrics; profiled end-to-end timings must not replace unprofiled benchmark timings.

CPU timings exclude initialization; GPU CUDA-event timings exclude allocation, initialization, compilation and final D2H. Wall timings additionally include final result D2H, not all cold-start work. CPU comparison is single-threaded (including serial multi-chain runs), not an optimized multicore baseline. The GPU had another graphics process throughout. All raw repetitions, including single-chain FP64 being slower than cached CPU, are retained in the final suite. No Tensor Core MFU is claimed.

## Muon on desk（2026-10-24）

已实现并实际测试 `src/muon_fit.cu`，复用同一 ROI 渲染、解析梯度、初始化与完整 FP64 验收。目标固定为 320×128“周舒畅”，1600 高斯 × 8 链。原参数 Adam、调参后 Adam、N×6 Muon、两个 N×2 Muon + 标量 Adam 都在同一可执行文件中测试。

参数为 `[x,y,log σx,log σy,θ,a]`，按 `[.15,.15,.025,.025,.025,1]` 归一化后使用 Muon。N×6 覆盖全部参数；分组方案分别正交化位置和尺度，角度/亮度用 Adam。Muon 为 momentum=.95、Nesterov、NS5、原始形状缩放、无权重衰减。依据的上游提交为 `f98f1cacc0263b04290753e32be8d498c1efc806`，源码与许可证见 `source/muon_upstream.py` 和 `source/muon_LICENSE`。这是高斯参数的实验性适配，非原作者推荐的神经网络隐藏权重使用场景。本次 NS5 用 FP32，没有测 BF16/FP16 或声称 Tensor Core 加速。

第一轮扫描 Adam 步长乘数 [.5,1,2,4]、Muon [.25,.5,1,2,4]；Muon 最优在网格下界，进一步补测 [.03125,.0625,.125,.25] 及余弦末值 [.05,0]。调参只用 seed 基数42的8条链。选定配置后按完整进程耗时校准步数并冻结，再交错三轮及三组新初值1042/2042/3042。完整进程包括 CUDA 初始化、优化、全图 FP64 重渲染、文件输出和退出，不含编译、SSH。保留 GPU 上已有图形工作负载。

| 方案 | 步数 | 完整进程中位数 | MSE 中位数 |
| --- | ---: | ---: | ---: |
| 原参数 Adam | 8000 | 3.731 s | 1.3354 |
| Adam 步长×2 | 8000 | 3.722 s | 1.3339 |
| Muon N×6，lr=.25 | 6912 | 3.638 s | 1.7797 |
| Muon 两个N×2，lr=.125 + Adam | 8032 | 3.765 s | 1.7999 |

两种 Muon 在三个未参与调参的种子上都输给 Adam。N×6 在这些新初值下实际用了4.10–4.22秒，比 Adam 的3.73–3.80秒更久；没有隐藏这个预算差异。Adam 步长翻倍也未在这三个种子上改善原配置，因此不修改现有默认值。结论只适用于测试过的目标、规模和配置，不是对 Muon 的普遍否定。

- `results/desk_muon_pilot/`、`results/desk_muon_refine/`：全部调参数据。
- `results/desk_muon_study/study.json`：完整逐次计时、硬件状态、配置、源码 SHA-256。
- `source/desk_muon_{pilot,refine,study}/`：各阶段实测代码快照。
- `src/check_muon.mjs`：实际 GPU 的 NS5 更新对照独立 CPU FP64 参考，最大相对 L2 误差 9.31e-7；包含两种分组、32/1600高斯、首步/多步。
- `src/check_muon_outputs.mjs`：29份正式结果使用原版 JavaScript 渲染器重算，要求逐像素一致、全图 MSE 差小于1e-9。
- 报告入口 `http://127.0.0.1:8766/report.html#desk-muon-results`。

在能够连接 desk 的 tmux 网络环境中依次运行（会重新生成对应实验数据）：

```sh
./run_muon_pilot.sh
./run_muon_refine.sh
./run_muon_study.sh
node src/check_muon.mjs
node src/check_muon_outputs.mjs
python3 src/finalize_report.py
```

当前原型的 N×6 单 block 共享内存实现限制 N≤1900，N×2 与 Adam 路径 N≤4096；网页原有参数上限与默认优化器不受此次 desk 实验影响。

## Adam 的 Nsight / CUDA Graph / WebGPU 加速（2026-10-24，后续）

最新推荐入口是 `run_adam_accelerated.sh`，仍使用 FP32、1600高斯×8链、320×128“周舒畅”。最终验证始终是全图、全部高斯、不裁剪的 FP64/Float64 渲染。

| desk 同轮三次中位数 | 完整进程 | GPU | 全图 MSE |
| --- | ---: | ---: | ---: |
| 原 Adam，8000步 | 3.681 s | 2.335 s | 1.3243 |
| FMA + warp布局 + 12线程验收，6σ | 2.560 s | 2.150 s | 1.3320 |
| 推荐：以上 + 4.5σ，8000步 | **1.999 s** | 1.578 s | **1.3304** |
| 更紧4σ，8000步 | 1.812 s | 1.403 s | 1.3489 |
| 4.5σ，11000步 | 2.676 s | 2.283 s | 1.2227 |

推荐方案完整进程快1.84×，GPU快1.48×。三组未调参的seed1042/2042/3042上，4.5σ完整耗时2.097/2.075/2.125秒，MSE1.3655/1.3736/1.3542；对应原Adam为1.3676/1.3875/1.3518。ROI裁剪是一项近似，4.5σ减少尾部计算，6σ包围框仍作为保守选项；3.5σ首轮明显损害质量，没有选用。不要将此误差结论直接外推到更密集或更大规模场景。

CUDA Graph 已在 `src/graph_fit.cu` 实现，`--graph-batch 32` 捕获32步，通过GPU计数器连续推进学习率、动量修正。1/32/128/512步都测过。第一阶段的6σ同配置正式消融中，原Adam3.735秒，单独Graph3.729秒，基本无提升；所有建图/实例化/上传成本计入完整时间。推荐配置默认不开Graph。

Nsight Systems 显示原Adam的 kernel/memset 区间覆盖约94.6%，这不是occupancy或MFU。CPU API耗时包括队列反压，不能当成纯启动耗时。Nsight Compute 提示非融合FP32乘加和执行不均匀，推动了FMA与warp布局实验。第一阶段优化后GPU从2.388降到2.189秒；并行验收从1.074降到0.143秒。ROI进一步压缩了GPU计算。所有正式性能结果不挂profiler、不锁频、不清缓存，保留其他GPU进程。

网页已移植ROI和多Worker验收，入口：`http://127.0.0.1:8766/adam_roi_lab.html`。每次新建引擎/管线，含编译、初始化、计算、读回、Worker启动和全Float64验收；不含页面加载和计时外串行交叉检查。先预热一次，再三轮轮换：

| 本机Apple M5 WebGPU | 总耗时中位数 | GPU | 验收 | MSE |
| --- | ---: | ---: | ---: | ---: |
| 6σ / 单Worker | 3.663 s | 3.198 s | 0.369 s | 1.3357 |
| 4.5σ / 单Worker | 2.696 s | 2.179 s | 0.374 s | 1.3413 |
| **4.5σ / 4 Workers** | **2.368 s** | **2.181 s** | **0.104 s** | **1.3413** |
| 4σ / 单Worker | 2.372 s | 1.914 s | 0.373 s | 1.3539 |

网页推荐配置快1.55×，MSE增加约0.42%。既有WGSL已用32线程、subgroups和128步批量提交，这些不算本次新增收益。四Worker按不重叠的行分区，每个像素仍累加全部高斯，逐像素与串行Float64一致。主报告原试跑默认6σ仍保留以便复现旧基线，新加速入口默认4.5σ/4 Workers/纯Adam。没有把CUDA的FMA编译开关、Graph重放或新的半精度收益冒充成WebGPU实测。

关键文件与复现：

- `results/adam_acceleration_summary.json`：最新desk/WebGPU汇总。
- `results/desk_graph/`：Nsight Systems原始`.nsys-rep`、SQLite、统计CSV；Nsight Compute CSV、版本和汇总。
- `results/desk_graph_study/`：第一阶段交错消融，34份Adam输出经原始渲染器校验；同批历史Muon数据保留，后续不再优化Muon。
- `results/desk_roi_tight_pilot/`、`results/desk_roi_tight_study/`：裁剪试探、正式28份输出和原始渲染器检查。
- `results/webgpu_roi_study.json`、`results/webgpu_roi_checks.json`：13次浏览器运行、4组完整代表输出、原版渲染器复核与并行验收检查。
- `source/desk_graph_{pilot,fma,warp,study}/`、`source/desk_roi_tight_{pilot,study}/`、`source/webgpu_roi/`：各阶段源码快照。报告构建校验SHA-256。
- 第一阶段时间预算外推得到的11840步实际约4.10秒，超出原约3.73秒预算；保留数据，但不把它写成同耗时质量提升。最新11000步明确作为不同迭代数的质量选项。

```sh
# 在已有tmux网络环境连接desk
./run_adam_accelerated.sh
# 完整实验顺序（重建结果会覆盖对应实验输出）
./profile_graph_baseline.sh
./run_graph_pilot.sh
./run_graph_fma.sh
./run_graph_warp.sh
./profile_graph_optimized.sh
./run_graph_study.sh
./run_roi_tight_pilot.sh
./run_roi_tight_study.sh
node src/check_graph_outputs.mjs
node src/check_roi_outputs.mjs
node src/check_webgpu_roi.mjs
python3 src/summarize_graph_profile.py
python3 src/finalize_report.py
```

网页实验通过实验室按钮运行，结果保存在页面DOM，可导出保存；当前报告使用的实测JSON已留存。WebGPU的命令缓冲区只能提交一次，无法直接照搬CUDA Graph重放；后续可研究更少的GPU读回、流水化timestamp读取，以及更好的像素分块，但当前没有把这些列为已实现收益。
