# MCMC splatting: extraction, fast Chinese fitting, and WebGPU precision lab

Open `report.html` for the final report and the **live WebGPU ROI + Adam fitter**. The default target is the exact 320×128 “周舒畅” bitmap used on desk. Default settings are 1,600 Gaussians, 8 chains, 10,000 ROI MCMC steps and 8,000 Adam steps. Start from seeded target-weighted random initialization; no prefit parameters are loaded.

Serve this directory with `python3 -m http.server 8766 --bind 127.0.0.1` and open `http://127.0.0.1:8766/report.html#playground`. The target, WGSL and UI code are embedded in the report. WebGPU availability still depends on the browser and secure-context support.

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
