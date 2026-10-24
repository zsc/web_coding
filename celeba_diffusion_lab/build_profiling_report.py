"""Local, portable report for measured PPU training acceleration."""
from pathlib import Path
import html, json

ROOT=Path(__file__).resolve().parent
d=json.loads((ROOT/'results/ppu_acceleration/summary.json').read_text())
labels={'eager':'原实现','graph':'训练 CUDA Graph','triton':'Triton RMSNorm','graph_triton':'Graph + Triton RMSNorm'}
names={'unet':'U-Net · 6.62M','large_jit':'JiT · 32.63M'}
rows=[]; checks=[]; profiles=[]
for kind,model in d['models'].items():
    for mode,v in model['modes'].items():
        lo,hi=v['paired_speedup_range']
        memory='Graph 私有池 · 不可直接比' if 'graph' in mode else f'{v["peak_allocated_mib"]/1024:.2f} GiB'
        rows.append(f'<tr><td>{names[kind]}</td><td>{labels[mode]}</td><td>{v["median_ms"]:.2f}</td><td>{v["paired_speedup_median"]:.2f}× <small>({lo:.2f}–{hi:.2f})</small></td><td>{memory}</td><td>{v["n"]}/5</td></tr>')
    for name,c in model['checks'].items():
        checks.append(f'<tr><td>{names[kind]} / {labels[c["optimization"]]}</td><td>{c["checks"]["first_clipped_gradient"]["relative_l2"]:.2e}</td><td>{c["checks"]["parameters"]["relative_l2"]:.2e}</td><td>{c["max_loss_relative"]:.2e}</td><td>{"通过" if c["passed"] else "失败"}</td></tr>')
for name,p in d['profiles'].items():
    profiles.append(f'<tr><td>{name}</td><td>{p["kernel_count"]//2:,}</td><td>{p["kernel_sum_ms"]/2:.2f}</td><td>{p["gap_within_kernel_span_ms"]/2:.2f}</td><td><a href="{p["path"]}">SQLite</a></td></tr>')
complete=all(m['complete'] for m in d['models'].values())
unet=d['models']['unet']['modes'].get('graph',{})
jit=d['models']['large_jit']['modes'].get('graph_triton',{})
proof=d.get('resume_verification')
page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PPU 训练加速 · ASight / Triton</title><style>
:root{--ink:#20333c;--muted:#5b707b;--line:#dce5e8;--teal:#167c80}*{box-sizing:border-box}body{margin:0;background:#f2f5f5;color:var(--ink);font:16px/1.7 system-ui,-apple-system,sans-serif}main{max-width:1120px;margin:auto;padding:40px 28px 80px}a{color:#126c79}nav{display:flex;gap:22px;flex-wrap:wrap;font-size:14px;margin-bottom:42px}.eyebrow{font-size:12px;letter-spacing:.13em;color:var(--teal);font-weight:700}h1{font-size:clamp(32px,5vw,50px);line-height:1.2;letter-spacing:-.03em;margin:15px 0 22px}h2{font-size:24px;margin:0 0 12px}h3{font-size:17px;margin:20px 0 8px}p{margin:12px 0}small,.caption{color:var(--muted);font-size:13px}.lead{font-size:19px;max-width:900px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}.metric{padding:22px;background:#e5eeee;border-radius:14px}.metric strong{display:block;font-size:32px;line-height:1.3;color:var(--teal)}section{background:white;border:1px solid var(--line);border-radius:16px;padding:28px;margin:24px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}td,th{padding:12px 13px;border-bottom:1px solid var(--line);text-align:left}th{font-size:12px;color:var(--muted)}img{width:100%;display:block;margin:20px 0}.cols{display:grid;grid-template-columns:1fr 1fr;gap:25px}.note{padding:14px 18px;background:#edf5f4;border-left:3px solid var(--teal);border-radius:4px}pre{padding:18px;background:#152d36;color:#e6f3f4;overflow:auto;border-radius:9px;font-size:13px;line-height:1.6}code{font-size:.9em}details{padding:14px 0}summary{cursor:pointer;font-weight:600}.pill{background:#dfebeb;color:#195d63;padding:4px 10px;border-radius:30px;font-size:12px}.links{display:flex;gap:16px;flex-wrap:wrap;font-size:14px}.footer{color:var(--muted);font-size:13px}@media(max-width:700px){main{padding:24px 14px}section{padding:20px 15px}.cols,.metrics{grid-template-columns:1fr}h1{font-size:34px}}
</style></head><body><main>
<nav><a href="report.html">← CelebA 实验主报告</a><a href="#speed">实测速度</a><a href="#trace">瓶颈</a><a href="#correctness">正确性</a><a href="#run">训练入口</a></nav>
<div class="eyebrow">RGB 64 × 64 · PPU-ZW810E · 2026-10-24</div><h1>把训练加速落实到<br>时间线与真实更新。</h1>
<p class="lead">先减少 CPU 启动空隙，再融合 profiler 显示的热点。U-Net 已有可直接续训的 Graph 入口；JiT 单独比较 Graph、Triton RMSNorm 及两者组合。两者均直接在 RGB pixel space 训练，模型结构、batch 与精度保持原配方。</p>
<span class="pill">__STATUS__</span>
<div class="metrics"><div class="metric"><strong>__USPEED__</strong>U-Net 配对吞吐提升</div><div class="metric"><strong>__JSPEED__</strong>JiT Graph + Triton 配对提升</div><div class="metric"><strong>__PROOF__</strong>U-Net 真实断点续训 100 步</div></div>
<section id="speed"><div class="eyebrow">01 / UNINSTRUMENTED TIMING</div><h2>完整训练 step，五轮同卡交错比较</h2><p>包括真实缓存取样、CPU→PPU 搬运、随机翻转和加噪、前向、反向、梯度裁剪、fused AdamW 与一份 EMA。每个进程预热 20 步，测量 5×20 步；每轮重建进程并交替顺序。所有候选都保持 batch 32。</p>
<img src="results/ppu_acceleration/training_speed.svg" alt="U-Net与JiT五轮完整训练耗时对照"><div class="scroll"><table><thead><tr><th>模型</th><th>实现</th><th>ms / step ↓</th><th>配对吞吐比 ↑</th><th>峰值 allocated</th><th>完成轮数</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p class="caption">ms 为五个进程中位数的中位数。吞吐比先在同轮内计算再取中位数，因此不一定等于表中两个耗时的比值；括号是五轮最小–最大值，不是置信区间。测速关闭 profiler，不含初始化、Graph 捕获、验证和存盘。这是单 PPU 测量，不能直接当成 4 卡 DDP 的提速；CPU / 节点为共享环境。厂商运行时的 allocated 统计未完整反映 Graph 私有池，故不比较 Graph 的显存数值、不声称节省显存。</p></section>
<section id="trace"><div class="eyebrow">02 / ASIGHT + PYTORCH</div><h2>目前的瓶颈在哪里</h2><div class="cols"><div><h3>U-Net：先解决启动，再看布局</h3><p>每步约 2,730 个 kernel。Graph 保留这些 kernel，减少逐个提交产生的间隙。ASight 中卷积约 10.5 ms、Copy / cast 约 9.0 ms；PyTorch 同 trace 关联表明复制同时包含 dtype 与 NCHW / NHWC 布局转换。</p><p>下一步最值得试的是保持 channels-last 的 GroupNorm / SiLU 路径，减少来回转换。当前没有实现它，也不把全部 Copy / cast 都归因于 GroupNorm。</p></div><div><h3>JiT：局部融合优先</h3><p>保留厂商 GEMM 和 Flash attention。只把残差流中 25 个无仿射 RMSNorm 的前向、反向替换为 Triton：FP32 归约、原 epsilon 与输入输出 dtype 不变；带权重的 Q / K norm 保持原样。</p><p>下一步可比较 RMSNorm + modulation、SwiGLU 融合；改变累加顺序或中间 FP16 舍入时必须重新核对梯度与训练轨迹。</p></div></div>
<img src="results/ppu_acceleration/kernel_breakdown.svg" alt="ASight各训练实现的kernel耗时分类"><div class="scroll"><table><thead><tr><th>采样</th><th>kernel / step</th><th>kernel 合计 ms / step</th><th>首末 kernel 间空隙 ms / step</th><th>证据</th></tr></thead><tbody>__PROFILES__</tbody></table></div>
<p class="caption">每份 ASight 只抓预热后 2 步。图中各类别按完整 kernel 名称互斥归类；原始名称保留在 JSON。Profiler 会扰动 CPU 启动与时间线，表中空隙不用于计算正式加速比，也不是 occupancy、算力或带宽利用率。JiT Graph + Triton 的带 profiler 样本出现约 44 ms / step 的额外空隙，而无 profiler 完整 step 约 45 ms；这份时间线不能代表正常运行的空隙，速度判断使用上面的独立五轮计时。</p><div class="links"><a href="results/ppu_profile/copy_attribution.json">复制的 shape / dtype / stride 归属</a><a href="results/ppu_acceleration/summary.json">完整统计与 kernel 名称</a></div></section>
<section id="correctness"><div class="eyebrow">03 / EQUIVALENCE BEFORE SPEED</div><h2>加速不靠改变训练任务</h2><p>Graph 只捕获前向、loss 和缩放后的反向；动态 GradScaler、溢出跳步、梯度裁剪与 fused AdamW 保持原路径。捕获时关闭 autocast 权重缓存，确保每步读取更新后的 FP32 参数。</p>
<div class="scroll"><table><thead><tr><th>候选</th><th>首步梯度相对 L2</th><th>20 步参数相对 L2</th><th>最大 loss 相对差</th><th>预设门槛</th></tr></thead><tbody>__CHECKS__</tbody></table></div>
<p class="caption">门槛在运行前设定：首步梯度相对 L2 &lt; 0.005，20 步参数与 EMA 相对 L2 &lt; 0.0001，loss 相对差 &lt; 0.005。测试强制一次 Inf，并将 scale 增长间隔设为 2，覆盖跳步和 scale 改变。Triton 改变浮点归约顺序，短测试通过不等于长期质量已验证。</p>
<div class="note"><b>真实续训核对：</b>从 U-Net 60k 的同一完整断点出发，原 trainer 和新 Graph trainer 各跑 100 步；模型、AdamW step / m / v、EMA 0.999 / 0.9999、scaler 与 CPU / CUDA / NumPy RNG __PROOFDETAIL__。还执行了相同验证、采样与 checkpoint 保存。</div>
<p><a href="results/ppu_acceleration/resume_verification/verification.json">查看完整断点核对结果</a> · <a href="src/verify_resumed_graph.py">复现脚本</a></p></section>
<section><div class="eyebrow">04 / TOOL RELIABILITY</div><h2>保留失败，界定工具能说明什么</h2><p>首轮 JiT ASight 子进程退出 SIGSEGV，没有生成 trace；随后一个 Triton 测速进程也退出 SIGSEGV，当时另一任务在运行 ASight。原因尚未证实。失败日志保留，JiT 全矩阵在其他 profiler 结束后重新执行；正式统计使用独立复测目录，不混入失败轮次。</p><p>本次使用系统厂商 PyTorch 2.9.0 / Triton 3.5.0 / ASight v2.0.0_20251231-4f7cd70。参考 <code>ssh air4 ~/d/know_ppu</code> 的采集边界、数值检查、真实 stride 与 AB/BA 方法，没有替换驱动或共享 SDK。__COUNTER__</p><div class="links"><a href="results/ppu_profile/large_jit/asys.log">首轮 ASight 失败</a><a href="results/ppu_acceleration/large_jit/r0_triton.log">首轮 Triton 测速失败</a><a href="results/ppu_profile/reference_sources.json">know_ppu 来源与哈希</a></div></section>
<section id="run"><div class="eyebrow">05 / REUSABLE TRAINING PATH</div><h2>加速实现已用于正式续训</h2><p><code>src/train_finetune_fast.py</code> 默认启用 Graph，继续保留两份 EMA、原学习率与断点格式。复制 checkpoint 和 training.json 到一个新输出目录，然后在已分配 PPU 的作业内运行；不要覆盖对照目录。</p><pre>__TRAIN_COMMAND__</pre><p class="caption">80k 仅为后续命令示例，本轮没有启动这项长训练。已实跑的是 60k→60.1k 的等价性检查。可加 <code>--acceleration eager</code> 使用原训练路径。Graph 捕获有一次性成本，并保留私有内存池；固定 shape 的长训练更适合它。JiT 的 Graph + Triton 路径已用于两组 16 PPU 续训并完成质量评估；训练量和 batch 与历史基线不同，不能把质量变化单独归因于 Triton。</p>
<details><summary>复现当前测速与采集</summary><pre>/usr/local/bin/python3 src/run_acceleration_suite.py \
  --variant unet \
  --weights results/rgb_flow_unet40k/ema_40000.pt \
  --out results/YOUR_PROFILE_RUN</pre><p>使用 sofy 申请 1 PPU / 4 CPU，设置项目 overlay 的 PYTHONPATH。复测时避免与其他 ASight 采集并行。所有命令、运行代码哈希和源代码快照都保留在远端结果目录。</p></details>
<p><a href="report.html#training-16ppu">四组 16 PPU 最终曲线、样本与成本</a> · <a href="src/distributed_acceleration.py">多卡 Graph、梯度与溢出同步实现</a>。最新多卡任务另有 12 步完整状态数值检查和 16 份副本一致性检查；这里的五轮单卡测速保持原 batch32 / 一份 EMA 口径。</p><div class="links"><a href="src/train_finetune_fast.py">加速训练入口</a><a href="src/training_acceleration.py">Graph 实现</a><a href="src/triton_rmsnorm.py">Triton 内核</a><a href="src/run_acceleration_suite.py">对照实验入口</a><a href="results/ppu_acceleration/tasks.json">任务记录</a></div></section>
<p class="footer">本地研究报告 · 没有改变 RGB 数据、网络宽深、patch 或采样质量协议。旧 checkpoint 与 trainer 保留。</p>
</main></body></html>'''
repls={'__ROWS__':''.join(rows),'__PROFILES__':''.join(profiles),'__CHECKS__':''.join(checks),
       '__STATUS__':'五轮复测与诊断采集完成' if complete else '实验进行中 · 只展示已保存结果',
       '__USPEED__':f'{unet.get("paired_speedup_median",1):.2f}×',
       '__JSPEED__':f'{jit["paired_speedup_median"]:.2f}×' if jit else '待测',
       '__PROOF__':'逐位一致' if proof and proof['bitwise_equal'] else '待验证',
       '__PROOFDETAIL__':'全部逐位一致' if proof and proof['bitwise_equal'] else '尚待验证',
       '__TRAIN_COMMAND__':(' '+chr(92)+'\n').join(['/usr/local/bin/python3 src/train_finetune_fast.py','  --variant unet --batch 32 --resume','  --finetune-lr 5e-5 --steps 80000','  --out results/YOUR_NEW_RUN']),
       '__COUNTER__':'本报告的瓶颈结论来自时间线，未使用硬件计数器。'}
counter=ROOT/'results/ppu_acceleration/counter_probe/result.json'
if counter.exists():
    c=json.loads(counter.read_text());repls['__COUNTER__']=html.escape(c['summary'])
if (ROOT/'results/rgb_accelerated80k/jobs.json').exists():
    page=page.replace('80k 仅为后续命令示例，本轮没有启动这项长训练。已实跑的是 60k→60.1k 的等价性检查。','两条 U-Net 分支已用 Graph 从 60k 续到 80k，各 1 PPU；70k、80k 双 EMA 评估均完成。后续四组 16 PPU 训练也已完成，累计各 512 万张次，见主报告最终对照。')
for k,v in repls.items():page=page.replace(k,v)
(ROOT/'profiling.html').write_text(page)
print(ROOT/'profiling.html')
