PPO double-pendulum GPU lab

Local report: report.html (standalone; opens without a server or internet)
Source: source/original.html, untouched GitHub blob; source/provenance.json
Extract: node src/extract.cjs
Reference baseline: node src/baseline.cjs

Remote machine verified: desk, RTX 2080 SUPER, Ryzen 9 3900X.
Existing interpreter: /home/zsc/miniconda3/envs/py312/bin/python
Torch 2.10.0+cu128; no new global environment installed.

Remote examples, from /home/zsc/ppo_gpu_lab:
  node src/baseline.cjs
  python src/validate.py
  python src/graph_check.py
  python src/diagnose.py
  python src/run.py --mode cpu --envs 64 --horizon 32 --minibatch 64 --iterations 7 --out cpu_matched
  python src/run.py --mode eager --envs 64 --horizon 32 --minibatch 64 --iterations 7 --out eager_matched
  python src/run.py --mode graph --envs 64 --horizon 32 --minibatch 64 --iterations 7 --out graph_matched_fused
  python src/run.py --mode graph --envs 256 --horizon 64 --minibatch 2048 --iterations 120 --seed 7 --evaluate --out graph_scaled_seed7

The default reproduces SOURCE dynamics, including its physically suspect gravity
and mass matrix. --physical selects explicitly DIFFERENT, absolute-angle rigid
rod equations. This is a separate task and must not be mixed into a same-task
performance or learning comparison.

The original HTML's Mat.map callback passes (value,column,rowArray), whereas
backward/Adam expect (value,row,column). original_core.cjs preserves that bug.
repaired_core.cjs fixes ONLY that callback for a runnable JS timing reference;
it still has original entropy/action/GAE semantics. Torch corrects those issues.
A legacy one-minibatch test compares original objective/Adam numerics separately.

The graph captures rollout + GAE + shuffled PPO minibatches + fused Adam.
No network enlargement, FP16, replay buffer, or skipping PPO epochs.
Latency timings synchronize CUDA before and after the entire iteration.
Compile/capture/setup is recorded separately. Warmup changes are discarded by
restoring model, environment, optimizer moments and RNG in place.
Evaluations are deterministic mean-action policies, 1000-step cap, fixed reset
seeds; evaluation time is excluded from reported training throughput.
JS runs use V8 Node on desk, with render/DOM/timers excluded; this is not a
browser wall-clock benchmark. Float64 JS versus Float32 torch are disclosed.
Machine had background GPU activity. Logs and measurements are retained.

bridge.py dispatches ssh/scp via a new tmux window because direct child-process
network access to desk is unavailable in this app context. It never sends keys
to the user's existing interactive SSH pane.

Re-run from the local directory: ./run_desk.sh
Rebuild HTML using saved JSON: python3 src/build_report.py
Independent exported-policy check: node src/independent_eval.cjs
Local artifacts are in results/desk/. Report selects seed 7 for performance,
with all training seeds retained. The full report data are results/summary.json.


SECOND OPTIMIZATION PASS (V2)
Recommended: python src/run.py --mode v2 --envs 256 --horizon 64 --minibatch 2048 --iterations 300 --seed 7 --physical --evaluate --out v2_run
All V1 modes remain supported. report_v1.html preserves the first report.

V2 keeps FP32, 819 parameters, PPO objectives, minibatches, epochs and episode
boundaries. It stores network weights and Adam moments in contiguous vectors,
compiles torch.func.grad_and_value together with the Adam update, generates
rollout random inputs in bulk, fuses the per-step actor/simulation operations,
and evaluates all current/next-state critic values in batches after rollout.
Critic parameters never change during a rollout; this reordering is equivalent.
Random-number consumption order differs from V1; fixed-noise tests verify
semantics, and 3 independent training seeds verify policy quality.

python src/validate_v2.py    # 20 Adam updates + boundary/reset rollout parity
python src/benchmark_v2.py   # 4 variants interleaved, 30 trials each
python src/profile_v2.py     # Remaining kernel profile

Measured paired workload: V1 35.826 ms, V2 17.031 ms, 2.104x faster.
V2 961,997 env steps/s includes all PPO updates. Setup/evaluation/I/O excluded.
Shared GPU, alternating forward/reverse order across all 4 variants.
Single-process ablation setup reuses compiled functions; do not treat its 0.19 s
as cold startup. Independent V2 training processes observed 9.85/3.12/3.43 s
setup, with existing partial compiler caches. A fresh compiler cache was not
benchmarked in this pass.

3 physical-model training seeds (7,11,23), 4,915,200 steps each, all survived
1000 steps in both 128-episode held-out reset sets and an independent 128-episode
Float64 JS simulator. This covers the narrow near-upright reset distribution;
it does not establish robustness to large disturbances or real hardware.

LIVE INTERACTION (historical first version; current defaults below)
Open play.html for the focused playground, or report.html for the full report.
Both embed all physics, controller code, model weights, and evaluation data.
No server, internet connection, GPU connection, or new training is needed to play.

Drag the cart or either circular joint. While held, physics pauses and the mouse
sets the pose; release zeroes all velocities and resumes if not manually paused.
This is pose placement, not a spring mouse constraint or velocity-based throw.
Buttons provide +/-0.35 m/s cart velocity kicks, a shared-angle starting pose,
pause/resume, reset, and automatic-control on/off. Focus canvas for arrow keys,
Space, and R. Drag angles are limited to +/-55 deg and cart placement to +/-2.2 m.
Episodes visibly stop at +/-60 deg or +/-2.4 m. No hidden reset or angle correction.

Original interactive-version default controller: discrete LQR, designed from the corrected physical equations
using autograd and scipy.linalg.solve_discrete_are. This is model-based feedback,
not reinforcement learning. That first interactive version did not retrain PPO; the WebGPU version below does.
The default 40 N motor is explicitly an interactive enhancement; 10 N reproduces
the original force limit. The old seed 7 PPO option uses its original 10 N limit and previously
trained v2_physical_seed7 weights. Motor enhancement changes the control task and
is not counted as PPO learning or throughput improvement. The interactive 60 deg
failure boundary also differs from the original training boundary of 12 deg.

Test: node src/recovery_eval.cjs
Rebuild both pages: python3 src/build_report.py
Design LQR on desk: /home/zsc/miniconda3/envs/py312/bin/python src/design_lqr.py

recovery_eval.json contains 8 signed angle combinations for each amplitude,
512 random states with independent angle +/-10 deg and position +/-0.25 m,
zero initial velocities, 10 s observation. Final 1 s must continuously satisfy
both angles <2 deg, angular speeds <0.15 rad/s, and cart speed <0.15 m/s.
Random test successes: PPO 10 N 191/512, LQR 10 N 363/512, LQR 40 N 512/512.
This is not a guarantee for arbitrary velocities, repeated disturbances, real
hardware, or swing-up from a fallen pose. Larger drags can fail in every mode.
1024 random one-step comparisons against an independent Gaussian matrix solve
had max absolute difference 1.11e-15; this checks solver implementation, not
external physical validity. Five demo scenarios (both poles, each pole alone,
cart placement, velocity kick) recover with the default controller.

WEBGPU TRAINING + DISTURBANCE CURRICULUM (current)
Double-click Open WebGPU.command to start localhost and open the training page.
Alternatively, open webgpu.html through localhost manually:
  cd /Users/zsc/Downloads/web_coding/ppo_gpu_lab
  python3 -m http.server 8778 --bind 127.0.0.1
  # http://127.0.0.1:8778/webgpu.html
The HTML embeds all shader code, Worker code, validation fixtures and weights.
No CDN, network service, new pip/npm dependency, or desk connection is required.
WebGPU must be supported by the browser. Saved-policy drag tests do not need it.

Connect GPU -> optional numeric validation / benchmark -> Start training.
Pause preserves model, environment, optimizer moments and RNG; Continue resumes.
Training finishes by selecting the fixed validation set's best checkpoint and
switching the live demo to its neural policy. Export saves full model and results.
An already measured seed 71 snapshot is embedded, so dragging works immediately.
report.html and play.html now also default to this new WebGPU-trained PPO snapshot.
LQR and the old 10 N PPO remain explicitly selectable reference controllers.

This is continued training from the earlier CUDA seed 7 weights, not from scratch.
No LQR teacher, behavior cloning, or hidden control fallback is used. The robust
mode changes force to 40 N, terminal angle to 60 deg and angle reward coefficient
to 4. It mixes 60% wide resets with 40% original near-upright resets. During the
first 200 iterations wide angles grow from +/-2 to +/-10 deg and kick amplitude
from zero to +/-0.35 m/s. Kicks occur with probability .003 per step after age 20.
Impulses are part of the transition: reward, bootstrap and next observation all
use the same perturbed state. Classic mode retains 10 N / 12 deg / original reward.
All modes use the corrected physical equations, FP32, 819 parameters, 256x64
rollouts, minibatch 2048 and 10 epochs (80 Adam updates per iteration).

GPU kernels: rollout, GAE, statistics/normalization, manual forward/backward,
parameter-gradient workgroup reductions and Adam. No CPU gradient fallback.
Weights, samples, activations and optimizer state stay GPU resident. CPU performs
Fisher-Yates shuffle and command encoding in a Worker. 3.81 MiB allocated buffers
include staging; this is not total browser process/GPU-driver memory.

Numeric reference: PyTorch 2.10 CPU FP32 autograd, 2048-sample clipping cases,
1 and 20 Adam updates; independent Float64 physics, GAE termination/truncation,
and cross-rollout value continuity with sampled impulses. Full checks and
thresholds are in results/webgpu_validation_worker.json.

Local speed measurement is Apple M5 / 10 GPU cores / Chrome 155, not desk's
RTX 2080 SUPER. results/webgpu_benchmark.json retains 30 timings after 5 warmups.
Wall time includes shuffle, uploads, command encoding, GPU submit/completion;
it excludes compilation, query readback and evaluation. GPU-only timestamps
cover all training kernels. Optional timestamp feature absence is supported.
No cold driver/shader cache benchmark was performed. Do not infer an API-only
WebGPU-versus-CUDA speedup from these different machines.

Three training seeds (71,79,83), 1200 iterations each, 19,660,800 environment
steps. Fixed validation seed 2026 / 128 episodes selects checkpoints. Browser
final test uses seed 20261025 / 512 episodes. After configuration freeze,
src/evaluate_webgpu.mjs audits all three on fresh seed 271828 / 1024 episodes
per scenario, including +/-10 deg poses and three velocity kicks. Early 600/1200
exploration is separate. Seed 71 is the prespecified display model. These are
simulator results and do not guarantee arbitrary drag recovery or swing-up.

Rebuild and audit from this lab directory:
  node src/evaluate_webgpu.mjs
  python3 src/build_webgpu.py
  python3 src/build_report.py
The builder assembles Worker code into a Blob script; no bundler is needed.
