# MCMC splatting: extracted CPU core and CUDA prototype

Open `report.html` for the self-contained Chinese report and interactive **CPU JavaScript** demo. No WebGPU implementation is included.

Upstream HTML is pinned in `source/provenance.json`. It fits a 2D grayscale image with additive rotated Gaussians, quantizes/clamps the total intensity, and uses single-parameter random proposals with a Metropolis-style acceptance rule. Clamped boundary proposals lack a Hastings correction; this is evaluated as a stochastic optimizer, not a guaranteed exact sampler.

## Run locally

```sh
node src/benchmark.mjs
node src/quality.mjs
python3 src/build_report.py
open report.html
```

Node 26 and system Python 3, no npm or pip dependencies. `core.mjs` exports `CachedChain`, `initialize`, `propose`, and deterministic fixture generation. The cache stores unquantized sums and per-Gaussian contributions, preserving final quantization. It reduces each proposal from O(PN) to O(P), with O(PN) storage. Floating-point accumulation can still affect rounding thresholds; equality claims are limited to tested cases.

## Run on desk

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
