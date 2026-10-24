CelebA 64x64 diffusion experiment (2026-10-24)

Open report.html for the Chinese report. Serve the parent directory if desired:
  python3 -m http.server 8770 --bind 127.0.0.1 --directory ..
  http://127.0.0.1:8770/celeba_diffusion_lab/report.html

Remote working directory:
  /home/zsc/celeba_diffusion_lab_20261024
Python:
  /home/zsc/miniconda3/envs/py312/bin/python
SSH uses the existing control connection:
  /tmp/mnist-desk-control (override with DESK_CONTROL_PATH)

The generator is trained from scratch: diffusers UNet2DModel, 6,338,499 parameters,
64x64 RGB, channels 32/64/128/128, two residual blocks per level, attention at 16x16
and in the 8x8 bottleneck. Linear DDPM schedule, T=1000, beta=1e-4..0.02,
epsilon MSE, AdamW lr=2e-4, 500-step warmup, clipping norm 1, warmup EMA <=0.999.
Batch 32; uniformly sample training images with replacement; random horizontal flip.
Weights stay FP32; forward/backward use CUDA FP16 autocast and GradScaler.

Data:
  Official list_eval_partition.txt MD5 d32c9cbf5e040fd4025c592c306e6668.
  Public mirror revision 2d31e6555722815c74ea7c845b07c1063dd705e9,
  https://huggingface.co/datasets/student/celebA
  Archive SHA256 9464985fa6cfc84d0fe6b87a6e4f07a275c8bd3dfc8abd713218765de7d62306.
  Mirror ZIP is repackaged, not byte-identical to the official ZIP.
  Verify all 202599 official filenames and original 178x218 image sizes.
  Center crop 178x178 at (0,20), Lanczos resize to 64x64 RGB.
  Train 162770, valid 19867, test 19962. Test images are not used to train or tune.
  src/prepare_data.py builds uint8 NPY caches and data/manifest.json.

Run:
  ./run_desk.sh sync
  ./run_desk.sh train                 # Fresh run; do not use on an existing run
  ./run_desk.sh status
  ./run_desk.sh resume 20000          # After a completed run: total target steps
  ./run_desk.sh evaluate              # Sampling and KID; run after training
  ./run_desk.sh fetch
  python3 build_report.py

Complete optimizer, AMP scaler, EMA, CPU/CUDA/NumPy RNG states are saved to
results/run1/checkpoint.pt on desk. Sampling previews preserve the training RNG
state. Intermediate EMA-only weights are saved at multiples of 5000 steps.
run_desk.sh fetch excludes checkpoints, image tensors and feature matrices;
these remain on desk. Human-readable results and PNGs are copied locally.

Validation:
  src/validate.py checks captured/eager FP32 trajectories, independent diffusers
  DDIMScheduler agreement, fixed seed repeatability, and stochastic graph outputs.
  src/evaluate.py repeats graph/eager comparison on the trained model.
  Seven randomly interleaved hot timing runs per sampler, batch=5, include final
  CPU copy and PNG encoding; exclude loading and graph setup. Shared GPU load.
  KID uses 1024 generated images and 2048 fixed validation images, 50 subsets of
  size 256. Polynomial kernel (x dot y / 2048 + 1)^3, unbiased squared MMD.
  Features: pytorch-fid 0.3.0, pool3, upstream pretrained Inception evaluator;
  evaluator weights are not used to train the generator.
  KID subset standard deviation is not a confidence interval. No full-dataset FID
  is claimed. Single-seed pilot; inspect all fixed-seed samples for defects.

Original project lineage:
  https://github.com/zsc/zsc.github.io/tree/0cca856e24d3d7de793d8d6ed0dc1e90db18174a/widgets/diffusion_demo
  The original algorithm extraction and MNIST optimization remain in ../diffusion_gpu_lab.

Grayscale distribution experiment:
  src/prepare_gray.py creates actual N x 64 x 64 x 1 uint8 caches using
  round(0.299 R + 0.587 G + 0.114 B), after the identical RGB crop/resize.
  src/train.py --grayscale --steps 5000 --batch 32 --out results/gray5k
  trains a fresh one-channel network with the same hidden layers and schedule.
  src/evaluate_gray.py --samples 512 compares actual gray-trained generation
  with RGB-trained 5000-step generation converted to gray for measurement.
  The latter is explicitly a separate comparator, not grayscale training.
  Every model is sampled with 100-step DDIM eta=0 and eta=1, fixed seeds
  881..912, batch 16. All 512 samples enter the statistics; the first 64 appear
  in each gallery. No image selection or brightness/color correction is used.

  Statistics use displayed uint8 values, 0..255: per-image spatial mean,
  per-image pixel SD, empirical CDFs, one-dimensional Wasserstein distances,
  and fraction outside real-image brightness percentiles 5..95.
  Reference: 2048 fixed evenly spaced validation images. A separate validation
  pool supplies 200 real-vs-real calibrations of 512 images; their 5..95 percentiles
  describe finite-sample variation, not a confidence interval for model quality.
  High-noise probes compare learned epsilon spatial-mean errors with a simple
  copy-noisy-mean baseline; they do not establish which network layer is causal.
  Distribution matching here does not mean Gaussianity or full-distribution equality.
  Gray training and RGB training have different channel counts and noise draws;
  their raw epsilon validation losses are not used as a direct quality ranking.

  ./run_desk.sh gray                 # Fresh run only; existing checkpoint rejects
  ./run_desk.sh gray-evaluate        # After gray 5000 steps complete
  # On desk, after gray evaluation:
  /home/zsc/miniconda3/envs/py312/bin/python src/plot_gray.py
  ./run_desk.sh fetch
  python3 build_report.py
  run_final_evaluation.sh is the sequential desk pipeline used for this report.

Color-loss control (separate from grayscale):
  run_color_ablation.sh forks the complete RGB 10000-step checkpoint twice.
  Both attempt 2000 further steps, with identical restored random streams.
  Control: epsilon MSE. Weighted: epsilon MSE + 256 * mean over batch of
  1[t>=800] * mean_channel(spatial_mean(epsilon_error)^2).
  Control makes 2000 successful updates; weighted makes 1992, skipping 8 for AMP
  overflow. The final NumPy/CPU/CUDA RNG states agree; see paired_rng_check.json.
  This weighted experiment failed to correct color drift and is retained as a
  negative result. It must not be presented as an improved model.

Measured grayscale outcome (this training seed and 5000-step budget):
  Real reference mean / across-image brightness SD / mean image contrast:
    112.61 / 35.50 / 61.15.
  Gray-trained eta=0: 130.40 / 76.13 / 47.95; brightness W1 41.52.
  Gray-trained eta=1: 127.07 / 60.88 / 50.46; brightness W1 26.64.
  Real-vs-real brightness W1 5..95 percentile range: 1.16..3.59.
  Gray training completed 5000 successful steps, zero AMP skips, 546.75 s.
  This is a failed distribution match, not an improved generator.
