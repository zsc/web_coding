# MCMC splatting: extraction, fast Chinese fitting, and WebGPU precision lab

Open `report.html` for the self-contained final report. Open `webgpu_lab.html` for the real **WebGPU / FP16** experiment (localhost recommended). The interactive area embedded in the main report uses CPU JavaScript in a background Worker. It accepts 1–10,000 Gaussians and 1–10,000,000 iterations, with live progress, stop, snapshot export, and a cache-size estimate. Defaults remain 100 Gaussians / 50,000 steps. At 10,000 Gaussians its FP64 contribution cache uses about 381.5 MiB; the 100×50 target size is unchanged. The Worker source is embedded, so the report remains self-contained.

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

## WebGPU / FP16

`src/webgpu.mjs` and self-contained `webgpu_lab.html` implement real device-resident MCMC in WGSL. On the local Apple M5 adapter, FP32 vs FP16-cache vs FP16-exp GPU medians were 116.33 / 101.84 / 101.97 ms for 160×64, 128 Gaussians, 4 chains, 8,192 steps; each has 3 interleaved repetitions after warmup. FP16 cache halves contribution memory (20 → 10 MiB), increasing throughput ~14.2%; FP16 exp adds no clear benefit. All modes use FP32 parameters/sums and integer SSE. Different arithmetic produces different trajectories; final loss and image error are recorded in `results/webgpu_precision.json`.

WGSL has optional f16 but no standard native bf16 scalar. BF16 emulation is not claimed as hardware acceleration. The final CUDA Adam pipeline is not yet ported to WebGPU; the 8.13-second quality result must not be represented as browser performance.

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
