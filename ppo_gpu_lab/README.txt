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
