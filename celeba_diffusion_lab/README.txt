CelebA 64x64 diffusion experiment (2026-10-24)

Completed RGB summary (evidence synced 2026-10-24 21:12 SGT)
---------------------------------------------------------------------
Open report.html for conclusions, final curves, all checkpoints and fixed
unfiltered samples; profiling.html separates implementation speedups from
hardware and batch-size effects. Root ../index.html is updated as well.

All four 1-node x16-PPU jobs SUCCEEDED, at5.12M cumulative image draws each:
  U-Net lr5e-5  task59156: slow-EMA KIDx1000=13.62 +/-1.77; wall5.90min.
  U-Net lr2e-4  task59152: slow-EMA KIDx1000=13.02 +/-1.84; wall6.66min.
  JiT   lr5e-5  task59155: slow-EMA KIDx1000=20.41 +/-2.33; wall6.35min.
  JiT   lr2e-4  task59153: slow-EMA KIDx1000=21.16 +/-2.38; wall5.85min.
KID uses2048 real/512 generated images,50 subsets of256; +/- is subset SD,
not a confidence interval. One training seed; no universal architecture claim.
Global batch512/local32; both models predict clean RGB pixels directly.
U-Net added3.2M draws from60k x32; JiT added2.56M from20k x128. Wall times
include successful-task checks/init/training/evaluation, but exclude prior
training, queueing and failed attempts. They are not from-scratch costs.
16 replicas of weights, dual EMA and optimizer matched exactly. Numerical
gates passed; all20 quality evaluations completed. Failed attempts retained.

Key evidence: results/rgb_16ppu/completion_summary.json, live_tasks.json,
per-run training.json/pipeline.json/numerical_gate.json/evaluation files.
Single-PPU matched small-JiT training:2.989x RTX2080SUPER. Paired single-PPU
implementation gains at batch32:U-Net Graph1.306x,largeJiT Graph+Triton1.147x.
Do not multiply these gains or treat the new16-card throughput as fixed-batch
scaling. Late loss improvements are not equal to KID improvements.

Rebuild local reports:
  .venv/bin/python src/plot_16ppu.py
  python3 build_report.py
  python3 build_profiling_report.py

The remaining sections retain historical recipes and reproducibility notes.

RGB / PPU environment and historical recipes
----------------------------------------------------------------------
Input and output are true RGB. Grayscale training has stopped; its report is
report_grayscale_archive.html. No new brightness/color ablations are running.

Remote shared project (reachable through air4 -> gpu3):
  /dataset-cpfs5/vtm-foundation-pretrain/temp/zhousc6/celeba_diffusion_lab_20261024
Sofy aliases live in isolated state on air4:
  ~/d/try_sofy/staging/celeba-diffusion-state
  celeba-1ppu: bifrost-2026102409500201-zhousc6 (old 1x2 cluster)
  celeba-4x16: bifrost-2026101312015311-zhousc6 (4 nodes x16 PPU)
remote_sofy.py targets sofy / sofy-large respectively; always specify the target.
Tunnels on air4 use localhost18771/18772; all jobs reserve their own cards.
Do not cancel entire clusters or modify other users' running processes.

Runtime: /usr/local/bin/python3 inside the PPU container; the default python3
is the Sofy environment and has no Torch. Keep vendor Torch2.9 and torchvision
unchanged. Export PYTHONPATH=$PWD/overlay and TORCH_HOME=$PWD/cache/torch;
overlay supplies diffusers0.29.0 and pytorch-fid0.3.0 without dependencies.

Small RGB control: JiT6,243,891 parameters, patch4, hidden192, depth9, heads6,
bottleneck32, conv4x64. 20k steps at batch32; retain5k checkpoint. U-Net flow:
6,624,899 parameters, original5k at batch32, completed40k; later forks are summarized above.
run_rgb_ppu.sh is the fresh-run entry point.
Both use data/train.npy and data/valid.npy (full hashes match desk exactly),
fixed[-1,1], random horizontal flip, x prediction, velocity-space MSE,
t=sigmoid(N(-.8,.8^2)), AdamW2e-4, FP16/scaler, EMA and Heun50/99NFE.

Larger RGB control: src/model_scaled.py,32,629,107 parameters; hidden384/depth12.
Patch4, heads6, bottleneck32 and conv4x64 unchanged. DDP4 PPU, globalbatch128,
10k steps (1.28M image draws). Save1250 and5000 for exposure matches with small
5k and20k (160k and640k draws). Same LR; warmup16000 image draws and EMA decay
adjusted by image count. Batch and EMA differ, so this is NOT capacity-only.
Full optimizer/model/scaler plus per-rank RNG snapshots are saved remotely;
train_scaled.py intentionally requires a fresh directory (no resume CLI yet).

Fresh larger run inside a reserved4-card task:
  /usr/local/bin/python3 -m torch.distributed.run --standalone --nproc_per_node=4     src/train_scaled.py --steps 10000 --global-batch 128 --out results/rgb_scaled_jit10k
For scaling: same script --benchmark --global-batch 128 with1/4/8 ranks.
Measured same-global-batch throughput: ~701 /2280 /2309 images/s.
8 cards offer only1.3% over4 here; formal training uses4 cards on one node.
Cross-node scaling is not tested. Replica agreement and finite loss passed.

RGB device comparison: src/benchmark_device.py --channels 3, identical5k EMA,
CPU-generated resident real inputs and code hashes. 1 PPU vs RTX2080SUPER:
2.99x training (33.14 vs99.05 ms/batch32),1.80x graph sampling (batch5,99NFE).
Vendor/runtime and host differ;
these are actual system measurements, not a peak-silicon FLOPS comparison.
See results/device_comparison_rgb.json and results/benchmark_rgb_*.

Evaluation (one reserved PPU, CPU BLAS threads4):
  python src/evaluate_rgb_flow.py
  python src/evaluate_scaled.py
  python src/record_rgb_environment.py
512 RGB samples per checkpoint, seeds881..912, show first64 unselected.
KID uses2048 fixed validation images, Inception pool3,50 subsets256. The shown
subset SD is not CI. Single training seed; no full-dataset FID claim.
watch_rgb_scaled_local.sh fetches small JSON/PNG artifacts and rebuildsreport.
It is a local process for this experiment, not a recurring automation.

Completed RGB results (KID x1000; lower is better):
  U-Net6.62M,160k image draws:46.25
  JiT6.24M,160k /640k draws:94.98 /40.01
  JiT32.63M,160k /640k /1280k draws:104.89 /45.24 /28.26
The larger model did not win at matched exposure in this single-seed run.
Its final improvement includes twice the exposure of the small20k control.
Final large training:573.96s wall clock,4 PPU,10000 successful updates,0 AMP
skips, replica checksum agreement. Final AMP-graph/FP32 trajectory RMSE3.20e-4.
Those original Sofy tasks completed successfully; the U-Net continuation
also completed; new round2 jobs below are running. Other users' jobs and cluster services remain untouched.

RGB U-Net continuation (user selected40k /1.28M image draws)
---------------------------------------------------------
Task62018 on celeba-1ppu: original src/train_flow.py,1PPU,batch32,lr2e-4.
Copied original5k checkpoint.pt and training.json to results/rgb_flow_unet40k,
then --variant unet --steps40000 --batch32 --resume. Full AdamW/EMA/scaler/RNG
restored; no recipe or source change. Original5k output is preserved.
Resume checkpoint SHA256:
20d421d1a36c478e4ac78db49b539b3e9dd61b77804cbd4dc7db5999d710c0c5

Task59100 tested1vs4 PPU with globalbatch32:45.918vs54.655ms/step.
4PPU is19% slower, so continuation remains single-card. DDP/serial FP32
gradient relativeL2=4.48e-5; no timed AMP skips. No DDP trainer was adopted.
Task62019 evaluates20k and40k on a separate reservedPPU using the same512
samples, Heun99NFE and2048real features as JiT. It checks20k RNG agreement
and FP16graph/FP32 trajectory differences. See src/evaluate_unet_continuation.py.
watch_unet_local.sh syncs PNG/JSON and regenerates curves/report every45s
until both evaluations finish. No recurring automation is installed.
U-Net20k is matched to smallJiT20k; U-Net40k is matched to largeJiT10k by
image exposure only (largeJiT uses different batch/EMA/capacity).

Historical experiments and original setup
----------------------------------------

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

U-Net affine ablation (A/B/C):
  A: gray5k, original additive time embedding, 6,337,345 parameters.
  B: time_affine5k, diffusers resnet_time_scale_shift=scale_shift, 6,597,153.
  C: global_affine5k, B plus image-derived continuous conditioning, 6,622,945.
  C concatenates 64*raw spatial mean, raw population SD, and the pooled output
  of Conv(1,16,3)-SiLU-Conv(16,32,3,stride2)-SiLU; MLP 34->128->128.
  The last MLP layer is zero-initialized. With the same seed, all B backbone
  parameters and B/C initial outputs match exactly. The condition is added to
  the time embedding through diffusers' identity class-embedding API, with no
  semantic labels and no clean-image statistics passed into the model.
  Each residual block predicts scale/shift from this combined embedding.
  This is not a full adaLN-Zero conversion: only C's extra condition starts zero.

  Fresh training commands on desk (from the remote experiment directory):
    python src/train.py --grayscale --variant time_affine --steps 5000 --batch 32 --out results/time_affine5k
    python src/train.py --grayscale --variant global_affine --steps 5000 --batch 32 --out results/global_affine5k
    python src/evaluate_affine.py
    python src/plot_affine.py
  Use /home/zsc/miniconda3/envs/py312/bin/python on desk.
  run_affine_experiment.sh runs these sequentially in tmux celeba-affine.
  --resume validates architecture and grayscale configuration before loading.
  Existing checkpoints reject accidental overwrite without --resume.

  src/validate_affine.py checks shared initialization, finite backward through
  the new branch, nonzero encoder gradients, input sensitivity, exact save/load,
  and FP32 CUDA Graph/eager agreement. Results: results/affine_validation.json.
  Evaluation uses the same 2048 validation references, seeds 881..912, 512 samples
  per method, DDIM 100 eta=0/1 as gray_evaluation; baseline measurements reused.
  All samples enter metrics, only the first 64 are shown. One training seed.
  Timing: 7 interleaved warm runs, batch5, DDIM100 eta0, AMP/Graph, CPU transfer
  included, PNG encoding excluded (unlike the older RGB timing protocol).

Local JiT references:
  source/reference_review.json records the exact local files and hashes read
  from ~/d/cstft_jit and ~/d/cstft_jit_arc2face. They were not modified or trained.
  Relevant differences: adaLN/RMSNorm, zero residual initialization, clean-x
  prediction with velocity-space flow loss, dataset-level reversible per-channel
  normalization, and optional post-unpatchify convolution for early grid artifacts.
  The references' unconditional adaLN uses time alone; the raw noisy-image global
  branch tested here is an additional hypothesis, not an existing JiT feature.
  cstft_jit's optional RPCA FiLM head is separate from its main CelebA configuration.
  The reference CelebA run uses ~130.4M parameters and 192x this experiment's image
  exposure; its reported quality is not an architecture-only comparison.

Requested JiT follow-up (paired clean-x flow):
  D: flow_unet5k, global-affine U-Net, 6,622,945 parameters.
  E: flow_jit5k, exact local-reference JiTSTFT implementation configured with
     depth9, hidden192, heads6, patch4x4 (256 tokens), patch projection middle32,
     and a 4-layer 3x3 residual output conv head, hidden64, last layer zero-init.
     6,234,385 parameters, including the conv head. User explicitly allowed conv.
  src/vendor_jit contains exact copies of the model/config and the sampler used
  for numerical reference, with source paths/hashes in source/jit_vendor_manifest.json.
  Original reference repositories remain unchanged. This is a small controlled
  experiment, not replication of the 130M/30k-step/global-batch1024 reference run.

  t=sigmoid(N(-0.8,0.8^2)); z=t*x+(1-t)*noise, noise_scale1, direct clean-x output.
  Loss=mean(((x_pred-x)/max(1-t,0.05))^2), algebraically the reference v-space MSE.
  Both use zero output initialization; U-Net t input is continuous t*1000.
  Same fixed [-1,1] data, augmentation, AdamW2e-4, 500-step warmup, EMA as A/B/C.
  Does NOT use JiT's dataset mean/std, original large-batch optimizer or EMA recipe.
  Epsilon-to-flow changes the whole denoising recipe (target/path/time/init/sampler),
  so its effect cannot be attributed solely to x prediction. D vs E holds it fixed.

  Heun50 intervals with last Euler = 99 NFE, compared with DDIM100 = 100 NFE.
  No intermediate clamp or image recentering. Raw sample tensors are saved and
  out-of-range pixel fractions reported before standard display quantization.
  Same seeds881..912 and 512 samples per method. Small-sample gray KID repeats
  gray to RGB identically for real/fake; 2048 real, 50 subsets256. Subset SD is
  not CI and scores are not directly comparable to earlier RGB KID.

  On desk: bash run_flow_experiment.sh
  It waits for tmux celeba-affine (including its benchmark) to finish, runs
  validate_flow.py, trains D then E, evaluates and plots. No concurrent GPU jobs
  within this task. Existing external desktop GPU processes are not modified.
  Fresh command examples:
    python src/train_flow.py --grayscale --variant unet --steps 5000 --batch 32 --out results/flow_unet5k
    python src/train_flow.py --grayscale --variant jit --steps 5000 --batch 32 --out results/flow_jit5k
    python src/evaluate_flow.py
    python src/plot_flow.py

Historical grayscale phase (stopped when RGB was restored): focus on quality, not more
brightness/color ablations. JiT + conv now runs continuously to 20,000 attempts
in results/flow_jit20k; a 5,000-attempt U-Net flow control follows. Training writes
EMA/RNG/training-metadata snapshots every 5,000 steps. Evaluation compares the
retained JiT5k checkpoint with U-Net5k, and separately JiT5k with JiT20k; never
attributes the latter's 4x training budget to architecture alone. No further DC
probes or eight-way color KID sweep are run. The three flow checkpoints receive
512-sample galleries and a three-way grayscale KID evaluation.
watch_training_local.sh updates the local report during this active run; it is
an ordinary local monitoring process, not a recurring scheduled automation.


U-Net40k final and newly queued follow-ups
---------------------------------------
U-Net40k:1985.60s cumulative wall,39987successful/13skipped,EMAval0.08863096.
RGB KIDx1000:5k46.25 ->20k23.48 ->40k19.08;40k matches largeJiT10k1.28M draws.
Tasks62018 and62019 SUCCEEDED. Main40k checkpoint is preserved separately.

Round2 on celeba-4x16,19 training PPUs total:
59105/59106/59107/59110:32.63M JiT,batch128/512 x LR2e-4/8e-4,2.56M draws each,
4PPU per run. Existing128/2e-4 resumes10k;other arms use same initialization.
59111:small RGB JiT20k->40k,1PPU,unchanged batch32/LR/EMA.
59112/59113:U-Net40k->60k,1PPU each,LR2e-4 versus5e-5;paired EMA0.999/0.9999.
Both EMA streams start from40k EMA;the slower stream is not a whole-run EMA.
59114/59115:interval guidance[.1,.8],scales1.3/1.6/2.0,20kstrong/5kweak models.
59116/59117:all16 checkpoint evaluations completed across2 shards using the fixed RGB protocol.
watch_round2_local.sh syncs results until both evaluation workers complete.
See results/round2_jobs.json for submission records; completed outcomes are retained in the report.
The referenced diffusion_demo2 sources and metrics remain unchanged.

PPU profiling and training acceleration (2026-10-24)
--------------------------------------------------
New local report: profiling.html; source: build_profiling_report.py.
Reference: ssh air4 ~/d/know_ppu (hashes in results/ppu_profile/reference_sources.json).
Isolated 1-PPU profiling tasks preserve historical model and trainer source.
src/profile_train.py: actual RGB CPU cache/H2D/augmentation + FP16 forward/backward, clip, fused AdamW, EMA.
src/run_acceleration_suite.py: numerical gate, then five fresh-process AB/BA rounds, followed by separate ASight captures.
src/training_acceleration.py: forward/loss/backward CUDA Graph, original GradScaler and optimizer path.
src/triton_rmsnorm.py: optional non-affine JiT RMSNorm only; no GEMM/attention replacement.
src/train_finetune_fast.py: opt-in U-Net continuation with graph (default), same two EMAs and complete checkpoint format.
Verified 60k->60.1k real continuation against historical trainer: model, both EMAs, AdamW states, scaler and all RNG states bitwise equal.
The graph loss tensor is cloned before collecting logging buckets to avoid retaining reused capture storage.
The profiling pass itself ran only equivalence checks. Subsequently both Graph80k continuations and four16-PPU forks completed; see the current summary above.
Timing excludes initialization/graph capture, validation and checkpoint writes; it includes one EMA, while the real continuation check includes both.
Original JiT profiler/timing SIGSEGV logs are preserved. A complete independent serial repeat is the reported JiT matrix.
Rebuild: python3 src/analyze_training_profiles.py; .venv/bin/python src/plot_training_acceleration.py; python3 build_profiling_report.py

Final five-round single-PPU RGB32 results: UNet48.81->36.76ms, median paired speedup1.306x; largeJiT52.21->45.01ms with Graph+Triton, paired1.147x. All measured timing steps had0 AMP skips. Five ASight SQLite traces are local. ASight Compute returned101 (device not ready); no valid hardware counters. Historical sources verified unchanged in results/ppu_acceleration/local_source_hashes.json.

Accelerated training completed after explicit user request (2026-10-24)
------------------------------------------------------------------
Sofy59140: UNet lr5e-5, full60k->80k, Graph, RGB, batch32,1PPU.
Sofy59141: UNet lr2e-4 control, full60k->80k, same settings,1PPU.
Both keep EMA0.999/0.9999 and evaluate both at70k and80k on the same allocated PPU.
Frozen code and source-checkpoint SHA256 are in each new output's pipeline.json.
Outputs: results/rgb_accelerated80k/{unet_lr5e5_graph,unet_lr2e4_graph}.
Local report section: report.html#accelerated-training; watch_accelerated_local.sh syncs bounded snapshots.
Prior60k checkpoints remain unchanged; new training starts from original60k, not the60.1k benchmark copy.

2026-10-24: user requested every experiment to use 1 node x16PPU.
New results/rgb_16ppu/jobs.json tracks four RGB64 forks (UNet and32.63M JiT, lr5e-5 and2e-4); each localbatch32/global512, target5.12M cumulative image draws.
New src/run_16ppu.py freezes sources and runs a16-rank numerical gate before training. Training graphs exclude collectives; scaled FP32 gradients are averaged before unscale and global clipping. Full optimizer/scaler/EMA are inherited; new per-rank RNG streams are explicit because batch/world changes.
Evaluation shards the same32 generation seeds among16 ranks and preserves original512sample KID protocol. Numerical failures and any retries remain in jobs.previous_attempts and *_attempt1 directories.
The original80k batch32 runs59140/59141 completed successfully and remain controls.
Completed results: report.html#training-16ppu. Bounded local sync: watch_16ppu_local.sh.
