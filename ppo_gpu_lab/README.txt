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

LIVE INTERACTION (local, offline)
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

Default controller: discrete LQR, designed from the corrected physical equations
using autograd and scipy.linalg.solve_discrete_are. This is model-based feedback,
not reinforcement learning. No PPO disturbance retraining was performed here.
The default 40 N motor is explicitly an interactive enhancement; 10 N reproduces
the original force limit. PPO always uses its original 10 N limit and previously
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
