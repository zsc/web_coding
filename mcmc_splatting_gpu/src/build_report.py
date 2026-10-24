#!/usr/bin/env python3
"""Build a self-contained local report from measured JSON; no third-party dependencies."""
import json, pathlib, re, html
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(p): return json.loads((ROOT/p).read_text())
local=read('results/local_benchmark.json')
quality=read('results/quality.json')
gpu={n:read('results/desk/'+n+'.json') for n in ['single_fp64','batch_fp64','single_fp32','batch_fp32']}
prov=read('source/provenance.json')
fit=read('results/desk/fit.json')
pgm=(ROOT/'results/desk/fit.pgm').read_bytes()
header=re.match(rb'P5\n\d+ \d+\n255\n',pgm).end()
fit['image']=list(pgm[header:])
cpu_rows=''.join('<tr><td>{W} × {H}<small>{N} 高斯 · {iterations:,} 步</small></td><td>{reference_ms:.1f} ms</td><td>{cached_ms:.1f} ms</td><td class="green">{speedup:.1f}×</td><td>{cache_MiB:.2f} MiB</td></tr>'.format(**c) for c in local['cases'])
gpu_rows=''.join('<tr><td>{dtype} · {chains} 链</td><td>{cpu_cached_all_ms:.1f} ms</td><td>{gpu_all_ms:.1f} ms</td><td class="green">{gpu_vs_cpu_cached_same_work:.2f}×</td><td>{gpu_total_steps_s:,.0f}</td></tr>'.format(**g) for g in gpu.values())
core=(ROOT/'src/core.mjs').read_text().replace('export ','')
worker=core+'\n'+re.sub(r'^import .*?;\n', '', (ROOT/'src/playground_worker.mjs').read_text(), count=1)
playground='const PLAYGROUND_WORKER_SOURCE='+json.dumps(worker,ensure_ascii=False).replace('<',r'\u003c')+';\n'+(ROOT/'src/playground.mjs').read_text()
data=json.dumps(dict(local=local,quality=quality,gpu=gpu,fit=fit),separators=(',',':')).replace('<',r'\u003c')
page=r'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MCMC Splatting · GPU 优化实测</title>
<style>
:root{--bg:#111616;--card:#1a2221;--line:#35443e;--text:#edf2e8;--muted:#a5b3ab;--lime:#c4ef89;--orange:#ffb77f}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}
main{max-width:1210px;margin:auto;padding:0 40px 60px}a{color:inherit;text-underline-offset:4px;text-decoration-color:#769174}button,input,select{font:inherit}
button{cursor:pointer}button:disabled{opacity:.4;cursor:default}header{padding:24px 0;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;font:11px/1.5 ui-monospace,monospace;letter-spacing:.12em}
nav{display:flex;gap:24px;color:var(--muted)}nav a{text-decoration:none}.brand:before{content:"";display:inline-block;background:var(--lime);width:7px;height:7px;border-radius:100%;margin-right:11px}
.hero{padding:60px 0 35px;display:grid;grid-template-columns:1.65fr 1fr;gap:30px;align-items:center}.eyebrow,.index{font:11px/1.5 ui-monospace,monospace;letter-spacing:.13em;color:var(--muted)}
h1,h2,h3,p{margin:0}h1{font-size:clamp(38px,4.6vw,62px);font-weight:600;line-height:1.25;letter-spacing:-.04em;margin:20px 0}h1 em{font-style:normal;color:var(--lime)}
h2{font-size:29px;font-weight:550;line-height:1.4;letter-spacing:-.02em}h3{font-size:17px;font-weight:550}.muted,.hero p{color:var(--muted)}.hero p{max-width:640px;font-size:15px}
.hero-art{position:relative;height:265px;display:grid;place-items:center;border:1px solid var(--line);border-radius:50%;background:radial-gradient(ellipse,#b3d98c12,transparent 70%)}
.hero-art:before,.hero-art:after{content:"";position:absolute;inset:70px 35px;border:1px solid #b4df7a77;border-radius:50%;transform:rotate(-25deg)}.hero-art:after{inset:85px 25px;transform:rotate(30deg);border-color:#ffb77f77}
.hero-art strong{font:30px Georgia,serif;z-index:1}.hero-art small{position:absolute;bottom:28px;color:var(--muted);font:10px ui-monospace,monospace}
.chips{display:flex;gap:7px;flex-wrap:wrap;margin-top:22px}.chips span{font:10px/1.7 ui-monospace,monospace;border:1px solid var(--line);border-radius:30px;padding:4px 9px;color:var(--muted)}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line);border-radius:13px;background:var(--card);overflow:hidden}
.metric{padding:26px}.metric+.metric{border-left:1px solid var(--line)}.number{font-size:49px;line-height:1.2;letter-spacing:-.05em;color:var(--lime);margin:8px 0 12px}.metric:last-child .number{color:var(--orange)}
.metric p{font-size:12px;color:var(--muted)}.note,.caption{font-size:12px;color:var(--muted);line-height:1.8}.caption{margin-top:15px}.section{padding-top:58px;scroll-margin-top:20px}
.section-head{display:flex;justify-content:space-between;align-items:flex-end;gap:30px;margin-bottom:23px}.section-head p{max-width:450px;color:var(--muted);font-size:13px}.index{color:var(--lime);margin-bottom:10px}
.gallery{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.image-card,.panel{border:1px solid var(--line);border-radius:12px;background:var(--card);padding:23px}
.image-card{padding:16px}.image-card h3{font-size:12px}.image-card canvas{display:block;background:black;width:100%;aspect-ratio:2;border-radius:5px;image-rendering:pixelated;margin:19px 0}.image-card strong{font:19px ui-monospace,monospace}.image-card p{font-size:11px;color:var(--muted);margin-top:8px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:17px}.formula{font:23px/1.8 Georgia,serif;color:var(--lime);margin:16px 0}.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:22px;border-top:1px solid var(--line);padding-top:24px;margin-top:24px}
.steps h3{font-size:14px;margin:7px 0}.steps p{font-size:12px;color:var(--muted)}.notice{margin-top:24px;padding:15px 20px;border-left:2px solid var(--orange);background:#ffb77f08;color:#d9c9b9;font-size:12px}
.tabs{display:flex;gap:10px;margin:20px 0}.tabs button,.secondary{padding:7px 15px;border:1px solid var(--line);border-radius:25px;color:var(--muted);background:transparent;font-size:12px}.tabs button.active{background:var(--lime);border-color:var(--lime);color:#162411}
.table-scroll{overflow:auto}table{width:100%;min-width:570px;text-align:left;border-collapse:collapse;font-size:12px;margin-top:10px}th{font-weight:400;color:var(--muted);font-size:11px}th,td{padding:14px 10px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}tr:last-child td{border:0}td small{display:block;color:var(--muted);font-size:10px}
.green{color:var(--lime)}.orange{color:var(--orange)}ul{padding-left:18px;color:var(--muted);font-size:13px}li{margin:9px 0}.chart{width:100%;height:240px;display:block;margin-top:20px}.legend{display:flex;gap:25px;font-size:12px;color:var(--muted)}.legend span:before{content:"";display:inline-block;width:16px;height:2px;margin-right:8px;vertical-align:middle;background:var(--orange)}.legend span:last-child:before{background:var(--lime)}
.code{background:#0c1110;border:1px solid var(--line);border-radius:8px;padding:17px;font:11px/1.9 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;color:#d0dfc8;margin:16px 0}
.live{display:grid;grid-template-columns:265px 1fr;gap:28px}.controls label{display:block;color:var(--muted);font-size:12px;margin-top:13px}.controls input,.controls select{width:100%;display:block;margin-top:5px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--text);padding:8px 10px}.controls input[type=file]{font-size:10px}
.primary{padding:11px 15px;border:0;border-radius:8px;background:var(--lime);color:#162511;font-size:13px;font-weight:600;margin-top:21px;width:100%}.controls .secondary{width:100%;margin-top:10px}
.live-pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.live-pair canvas{width:100%;aspect-ratio:2;image-rendering:pixelated;background:black;border:1px solid var(--line);border-radius:7px}.live-pair p{font-size:11px;color:var(--muted);margin-bottom:8px}.live-stats{display:flex;flex-wrap:wrap;gap:24px;margin:24px 0;font-size:11px;color:var(--muted)}.live-stats b{font:19px/1.5 ui-monospace,monospace;display:block;color:var(--text)}#live-status{font-size:12px;color:var(--lime);min-height:28px}
.files{display:grid;grid-template-columns:1fr 1fr;gap:8px 24px;margin-top:22px}.files a{padding:10px 0;border-bottom:1px solid var(--line);font:12px/1.8 ui-monospace,monospace}
details{margin-top:22px;padding-top:15px;border-top:1px solid var(--line)}summary{cursor:pointer;font-size:13px;color:var(--muted)}footer{margin-top:60px;border-top:1px solid var(--line);padding-top:20px;display:flex;justify-content:space-between;color:var(--muted);font-size:11px}[hidden]{display:none!important}
@media(max-width:800px){main{padding:0 20px 40px}nav{gap:12px;font-size:10px}.hero{grid-template-columns:1fr;padding-top:40px}.hero-art{display:none}.metrics{grid-template-columns:1fr}.metric+.metric{border-left:0;border-top:1px solid var(--line)}.number{font-size:38px}.gallery{grid-template-columns:1fr 1fr}.two,.live{grid-template-columns:1fr}.steps{grid-template-columns:1fr 1fr}.section-head{display:block}.section-head p{margin-top:15px}.files{grid-template-columns:1fr}.live-pair{gap:9px}footer{display:block}}
</style></head><body><main>
<header><div class="brand">MCMC / ACCELERATION NOTES</div><nav><a href="#results">实测</a><a href="#algorithm">算法</a><a href="#playground">试跑</a><a href="#reproduce">复现</a></nav></header>
<section class="hero"><div><div class="eyebrow">LOCAL STUDY · 24 OCTOBER 2026 · VERIFIED ON DESK</div>
<h1>一次只改一颗，<br><em>何必重画全部。</em></h1><p>从浏览器演示提取核心算法：缓存高斯贡献，让每次提案只做增量更新；再让多条独立链在 CUDA 内核中持续迭代。这里是代码、实测和可以直接运行的结果。</p>
<div class="chips"><span>Apple M5 / JS CPU</span><span>RTX 2080 SUPER / CUDA</span><span>自包含 HTML / 离线可用</span></div></div><div class="hero-art"><strong>S′ = S − g + g′</strong><small>ONE PROPOSAL. ONE GAUSSIAN.</small></div></section>
<div class="metrics"><div class="metric"><div class="eyebrow">CPU · 原算法优化</div><div class="number">{{cpu_gain}}×</div><p>100 × 50 / 25 高斯 / 单链<br>缓存 JS 对比原 HTML 的 JS 核心</p></div>
<div class="metric"><div class="eyebrow">GPU · FP64 验证通过</div><div class="number">{{gpu64_gain}}×</div><p>32 链 GPU 对比 CPU 串行完成同量工作<br>每链 1,200 步 · 同一台 desk</p></div>
<div class="metric"><div class="eyebrow">GPU · FP32 近似模式</div><div class="number">{{gpu32_gain}}×</div><p>32 链 GPU 对比 CPU 串行完成同量工作<br>存在数值差异，单独列出</p></div></div>
<p class="caption">三个数字基线不同，不能相乘。多链吞吐提升不等于一条链收敛快同样倍数。CPU 基线均为单线程。GPU 测试时保留了其他进程：初始约 39% GPU 利用率、约 1.6 GiB 显存占用，因此结果不是空闲 GPU 上限。</p>

<section class="section" id="output"><div class="section-head"><div><div class="index">01 / THE OUTPUT</div><h2>先看真正拟合出来的图。</h2></div><p>基准使用确定性的 MCMC 位图，消除字体差异。降温、多高斯、多链选优是质量策略，与等价计算加速分开评价。</p></div>
<div class="gallery">
<div class="image-card"><h3>目标 / 固定输入</h3><canvas id="target-image"></canvas><strong>100 × 50</strong><p>2 × 2 抗锯齿采样<br>灰度范围 0–255</p></div>
<div class="image-card"><h3>原设置 / 固定 T = 50</h3><canvas id="fixed-image"></canvas><strong>{{fixed_loss}} MSE</strong><p>25 高斯 · 10,000 步<br>缓存版 · 同原损失与接受规则</p></div>
<div class="image-card"><h3>CPU / 降温与增容</h3><canvas id="quality-image"></canvas><strong>{{quality_loss}} MSE</strong><p>100 高斯 · 100,000 步<br>T: 50 → 0.05 · {{quality_sec}} s（含校验）</p></div>
<div class="image-card"><h3>GPU / 32 链选优</h3><canvas id="gpu-image"></canvas><strong>{{fit_loss}} MSE</strong><p>100 高斯 · 每链 100,000 步<br>{{fit_sec}} s · FP32 · T: 50 → 0.05</p></div></div>
<p class="caption">展示最终状态。GPU 从 32 条链最终参数的 FP64 重渲染结果中选出最低 MSE。上列预算不同，不能直接用质量图计算公平加速比；GPU 不保证找到全局最优。</p>
<div class="panel" style="margin-top:18px"><div class="legend"><span>固定 T=50 / 25 高斯</span><span>降温 50 → 0.05 / 25 高斯</span></div><canvas id="loss-chart" class="chart" width="1040" height="260"></canvas>
<p class="caption">质量消融：相同初值、随机数序列、高斯数与 10,000 步预算，最终 MSE 从 {{fixed_loss}} 降至 {{annealed_loss}}。这是固定 seed 的单次观察，不是多 seed 统计。</p></div></section>

<section class="section" id="algorithm"><div class="section-head"><div><div class="index">02 / THE CORE ALGORITHM</div><h2>它是 2D 加法高斯拟合。</h2></div><p>没有 3D 相机、深度排序或 alpha 合成。每颗高斯只有位置、两个尺度、角度和幅度这六个参数。</p></div>
<div class="two"><div class="panel"><h3>保留下来的计算定义</h3><div class="formula">gⱼ(p) = αⱼ exp[−½(u²/sₓ² + v²/sᵧ²)]<br>I(p) = clip(round(Σⱼ gⱼ(p)), 0, 255)</div><p class="note">u、v 为旋转到高斯局部坐标后的像素位置。损失 L 是 I 与目标的全图 MSE；取整和截断在求和以后进行，不能先截断 S 再做增量更新。</p></div>
<div class="panel"><h3>真正要消掉的浪费</h3><div class="formula">S′(p) = S(p) − gₒₗd(p) + gₙₑw(p)</div><p class="note">原版每次只改一个参数，却重算所有像素上的所有高斯，重复三角函数并 JSON 深拷贝整个数组。缓存总和与单颗贡献，候选只算一颗：每步从 O(PN) 降至 O(P)，贡献缓存占 O(PN) 内存。</p></div></div>
<div class="steps"><div><div class="index">01 / PROPOSE</div><h3>选择高斯与参数</h3><p>均匀扰动并保留原 clamp。位置步长 2，尺度 1，角度 0.1，幅度 10。</p></div><div><div class="index">02 / EVALUATE</div><h3>只求新增贡献</h3><p>sin/cos 每颗预计算；复用候选缓冲，仍评价全部像素，不截断尾部。</p></div><div><div class="index">03 / ACCEPT</div><h3>按原规则接受</h3><p>min(1, exp(−ΔL/T))。损失是取整后图像的 MSE，SSE 用整数累加。</p></div><div><div class="index">04 / COMMIT</div><h3>接受后更新缓存</h3><p>CPU 交换数组；GPU 在 block 内统一决策、同步提交，再开始下一步。</p></div></div>
<div class="notice">边界说明：clamp 提案在参数边界通常不对称，原代码没有 Hastings 修正，因此这里把它视为随机优化器，不保证严格分布采样。增量式是实数代数等价；浮点累积在半整数取整边界可能改变轨迹，一致性结论只覆盖已测用例。</div></section>

<section class="section" id="results"><div class="section-head"><div><div class="index">03 / MEASURED, NOT ESTIMATED</div><h2>同时看基线、精度和工作量。</h2></div><p>所有数值来自保留的 JSON 日志。取 3 次计时中位数，初始化、预热、编译不混入迭代时间。</p></div>
<div class="tabs"><button class="active" data-panel="cpu-table">Mac / JavaScript</button><button data-panel="gpu-table">desk / CUDA</button></div>
<div class="panel result-panel" id="cpu-table"><h3>原始渲染函数 vs 缓存 · Apple M5</h3><div class="table-scroll"><table><thead><tr><th>规模</th><th>原始核心</th><th>缓存核心</th><th>加速</th><th>贡献矩阵</th></tr></thead><tbody>{{cpu_rows}}</tbody></table></div><p class="caption">Node v26.12.1，同一 JS realm，原渲染与 loss 函数从 HTML 直接提取。包含提案、接受与原基线的 JSON 深拷贝；均排除 DOM、绘制、rAF。各预热 60 步，3 次测试交错顺序。表中内存不含 sum / candidate 等小缓冲。</p></div>
<div class="panel result-panel" id="gpu-table" hidden><h3>同一台 desk · Ryzen 9 3900X + RTX 2080 SUPER</h3><div class="table-scroll"><table><thead><tr><th>模式 / 相同总工作</th><th>CPU 缓存串行全部链</th><th>GPU 全部链</th><th>加速</th><th>总 proposals/s</th></tr></thead><tbody>{{gpu_rows}}</tbody></table></div><p class="caption">每链 100 × 50、25 高斯、1,200 步、T=50。CPU/GPU 相同初值和随机流。CUDA event 计时且等待完成；256 步一批；FP64 / FP32 与各自同 dtype CPU 比较。CPU 未做多核并行。完整样本、含最终回传的 wall_ms 与 setup_ms 在 JSON 中。</p></div>
<div class="two" style="margin-top:18px"><div class="panel"><h3>验证通过的范围</h3><ul><li>JS：4 组边界、饱和、最小尺度、T=0 用例，各 500 步，接受决策、MSE、最终像素一致。</li><li>JS：3 组性能用例最终损失 / 接受数一致；长运行每 5,000 步完整重渲染检查并重建缓存。</li><li>CUDA FP64：前 500 步逐步比较完整重渲染 CPU 参考，决策和 SSE 全部一致；1,200 步后 32 链损失和接受数一致。</li><li>GPU 导出从最终参数用 FP64 重渲染，32 链最大缓存 MSE 漂移 {{fit_drift}}。</li></ul></div>
<div class="panel"><h3>不能过度解读的部分</h3><ul><li>单链 FP64 GPU / 缓存 CPU = {{single64_gain}}×，GPU 更慢；一个 block 只占一个 SM。</li><li>FP32 前 500 步相对完整 FP64 参考有 {{fp32_mismatches}} 次 MSE 不一致（决策仍一致），最大差 {{fp32_error}}；32 链最终有 {{fp32_final_mismatches}} 条与 CPU FP32 缓存状态摘要不同。</li><li>FP32 的 {{gpu32_gain}}× 是近似模式，多链的速度不能作为单链收敛速度。</li><li>GPU 测试有并发负载；目前没有大分辨率、多核 CPU 或多个输入的 GPU 性能结论。</li></ul></div></div></section>

<section class="section" id="gpu-design"><div class="section-head"><div><div class="index">04 / GPU DESIGN & UTILIZATION</div><h2>让状态和迭代留在显存里。</h2></div><p>网页试跑是 JS CPU，远端是 CUDA。当前没有 WebGPU 实现。网页使用 GPU 需要另写 WGSL compute shader 并验证。</p></div>
<div class="two"><div class="panel"><h3>已实现的并行布局</h3><pre class="code">GPU grid: 32 independent chains
  each block: 256 threads
    thread 0    → proposal + acceptance decision
    all threads → pixels + integer SSE reduction
    barrier     → commit before next proposal

Parameters / contributions / sum / RNG stay on GPU.
Host launches 256-step chunks; no per-step readback.</pre><p class="note">一条链仍严格顺序执行。不能从同一旧状态提出多个重叠候选后全部接受。工作区约 C × P × (N+2) × dtype 字节，启动限制不超过当时空闲显存的一半。</p></div>
<div class="panel"><h3>MFU：统计口径与实测状态</h3><div id="mfu-text">{{mfu_html}}</div><p class="note" style="margin-top:15px">nvidia-smi 的 GPU Util 是采样期内的忙碌比例，不是峰值 FLOPS 利用率，还包含其他进程。这个内核含 exp、除法、整数 SSE、同步，不能用“原始全量重算 FLOPs”作优化后的 MFU 分子；省掉的计算不算已执行 FLOPs。</p></div></div>
<div class="notice">下一步：大图可以研究单链跨多个 block + CUDA Graph；大量小图 / 独立初始化适合更多链。ROI 裁剪、快速 exp、Adam、FP32 边界回退和多核 CPU 均需要单独做精度 / 性能消融，当前未声称实现。</div></section>

<section class="section" id="playground"><div class="section-head"><div><div class="index">05 / TRY IT LOCALLY · CPU BACKEND</div><h2>这份报告里，也能直接跑。</h2></div><p>完全离线、无上传的 JS 缓存核心。改文字或载入图片，下载当前拟合与参数；目标可以导出为 CUDA 支持的 PGM。</p></div>
<div class="panel live"><div class="controls"><label>目标文字<input id="demo-text" value="MCMC" maxlength="20"></label><label>高斯数量 · 上限 10,000<input type="number" id="demo-N" min="1" max="10000" step="1" value="100" list="gaussian-presets" required></label><datalist id="gaussian-presets"><option value="25"><option value="100"><option value="800"><option value="1600"><option value="3200"><option value="5000"><option value="10000"></datalist>
<label>迭代数 · 上限 10,000,000<input type="number" id="demo-steps" min="1" max="10000000" step="1" value="50000" list="iteration-presets" required></label><datalist id="iteration-presets"><option value="10000"><option value="100000"><option value="500000"><option value="1000000"><option value="5000000"><option value="10000000"></datalist><p class="note" id="demo-budget" style="margin-top:10px"></p>
<label>温度<select id="demo-mode"><option value="annealed">降温 50 → 0.05</option><option value="fixed">固定 50 · 原设置</option></select></label><label>或载入图片<input type="file" id="demo-file" accept="image/*"></label>
<button class="primary" id="demo-start">初始化并运行</button><button class="secondary" id="demo-stop" disabled>停止</button><button class="secondary" id="demo-export" disabled>下载当前图像与参数</button><button class="secondary" id="target-export">下载目标 PGM</button></div>
<div><div class="live-pair"><div><p>TARGET / 100 × 50</p><canvas id="live-target" width="100" height="50"></canvas></div><div><p>INCREMENTAL FIT / JS CPU</p><canvas id="live-image" width="100" height="50"></canvas></div></div>
<div class="live-stats"><span>迭代<b id="live-iter">0</b></span><span>MSE<b id="live-loss">—</b></span><span>接受率<b id="live-accept">—</b></span><span>计算耗时<b id="live-time">—</b></span></div>
<p id="live-status">准备就绪。默认目标与基准位图相同。</p><p class="note">seed 固定为 42 / 101。初始化、迭代与缓存重建均在后台 Worker 中执行；每约 100 ms 更新进度。计算时间不含初始化、消息传输和调度等待。每 5,000 步重建缓存。图片缩放到 100 × 50 并转灰度；修改文字后使用浏览器字体，所以不与原基准位图直接比较。</p><p class="note" style="margin-top:15px">下载的是点击时的当前状态；点击“初始化并运行”会重新开始。本区不连接 desk，不调用 WebGPU。</p></div></div></section>

<section class="section" id="reproduce"><div class="section-head"><div><div class="index">06 / REPRODUCE & INSPECT</div><h2>所有结果，都留有来路。</h2></div><p>本地：mcmc_splatting_gpu<br>desk：~/mcmc_splatting_gpu_20261024</p></div>
<div class="two"><div class="panel"><h3>本地基准与报告</h3><pre class="code">cd /Users/zsc/Downloads/web_coding/mcmc_splatting_gpu
node src/benchmark.mjs
node src/quality.mjs
python3 src/build_report.py
open report.html</pre><p class="note">无 npm 依赖。报告所有数据内嵌，直接 file:// 打开即可。保留目录结构以访问源文件链接。原版 T=0 的 NaN 行为保留在 CPU 参考，交互温度限定为正数。</p></div>
<div class="panel"><h3>远端 CUDA</h3><pre class="code">./run_desk.sh
./run_fit.sh
# 以下在 desk 项目目录内运行：
./desk_benchmark --chains 32
./desk_benchmark --fit --fp32 --chains 32 \
  --gaussians 100 --steps 100000 --output results/fit
# 把下载的 target.pgm 复制到 desk 后：
./desk_benchmark --fit --fp32 --input target.pgm \
  --chains 32 --gaussians 100 --steps 100000 \
  --final-temperature 0.05 --output results/custom</pre><p class="note">编译：CUDA 12.8，-O3 -std=c++17 -arch=sm_75 --fmad=false。默认 FP64；--fp32 显式启用近似模式。输出目录须存在。最初直接 SSH 不通，沿用现有 tmux 进程环境后成功。</p></div></div>
<div class="files"><a href="src/core.mjs">core.mjs / 可复用缓存核心</a><a href="src/desk_benchmark.cu">desk_benchmark.cu / CUDA 与 CPU 参考</a><a href="source/mcmc_splatting.original.html">original.html / 原版快照</a><a href="results/local_benchmark.json">local_benchmark.json / 本地实测</a><a href="results/desk/batch_fp64.json">batch_fp64.json / FP64 验证与计时</a><a href="results/desk/batch_fp32.json">batch_fp32.json / FP32 误差与计时</a><a href="results/quality.json">quality.json / 降温消融</a><a href="results/desk/fit.json">fit.json / GPU 最终参数</a></div>
<details><summary>来源、方法细节与完整证据</summary><div class="note" style="margin-top:15px">
<p>固定源码：<a href="{{source_url}}">zsc/widgets · {{commit_short}}</a>。SHA-256：<span style="overflow-wrap:anywhere">{{sha256}}</span>。提取 renderer 与 loss 的本地参考文件为 src/reference.mjs。</p>
<p>参考：<a href="https://docs.nvidia.com/cuda/archive/13.2.2/cuda-programming-guide/01-introduction/programming-model.html">NVIDIA CUDA 编程模型</a>、<a href="https://developer.nvidia.com/blog/cuda-graphs/">CUDA Graphs 与提交开销</a>、<a href="https://docs.nvidia.com/nsight-compute/ProfilingGuide/">Nsight Compute 指标定义</a>。</p>
<p>远端硬件：Ryzen 9 3900X，24 逻辑 CPU；RTX 2080 SUPER，48 SM、compute capability 7.5、8 GiB；驱动 580.178.04。CUDA event 只统计迭代和批次调度，wall_ms 额外包含最终结果回传；setup_ms 在已创建 CUDA context 后统计分配和初始化，不是进程冷启动。GPU 三轮顺序固定；JS 三轮交错。测试日期以 Asia/Singapore 2026-10-24 标注。</p>
<p>FP32 拟合 workspace {{fit_memory}} MiB，总工作 32 × 100,000 次提案，耗时 {{fit_sec}} s；输出经最终参数 FP64 重渲染选优。不在 GPU 基准中混用精度或更少迭代来计算加速。</p></div></details></section>
<footer><span>LOCAL COMPUTE STUDY / 2026.10.24<br>源码 · 数值检查 · 实测 · 可复现</span><span>无 CDN · 无上传 · 无追踪<br>报告中的网页运行后端：CPU JavaScript</span></footer>
</main><script type="module">
{{core}}
const DATA={{data}};
const el=id=>document.getElementById(id);
function paint(id,a,W=100,H=50){const c=typeof id==='string'?el(id):id;c.width=W;c.height=H;const ctx=c.getContext('2d'),im=ctx.createImageData(W,H);a.forEach((v,p)=>{im.data[p*4]=im.data[p*4+1]=im.data[p*4+2]=v;im.data[p*4+3]=255;});ctx.putImageData(im,0,0);}
paint('target-image',DATA.local.demo.target);paint('fixed-image',DATA.quality.runs[0].image);paint('quality-image',DATA.quality.runs[2].image);paint('gpu-image',DATA.fit.image);
const chart=el('loss-chart'),ctx=chart.getContext('2d'),L=55,R=20,T=20,B=35;
ctx.font='11px monospace';ctx.fillStyle='#a5b3ab';ctx.strokeStyle='#35443e';
for(let v=0;v<=20000;v+=5000){const y=T+(260-T-B)*(1-v/20000);ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(1040-R,y);ctx.stroke();ctx.fillText(v/1000+'k',10,y+4);}
for(let v=0;v<=10000;v+=2000)ctx.fillText(v/1000+'k',L+v/10000*(1040-L-R)-5,250);
ctx.fillText('MSE',10,12);
DATA.quality.runs.slice(0,2).forEach((run,i)=>{ctx.strokeStyle=['#ffb77f','#c4ef89'][i];ctx.lineWidth=2;ctx.beginPath();run.curve.forEach((p,j)=>{const x=L+p.iteration/10000*(1040-L-R),y=T+(1-p.loss/20000)*(260-T-B);j?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();});
document.querySelectorAll('[data-panel]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-panel]').forEach(b=>b.classList.toggle('active',b===button));document.querySelectorAll('.result-panel').forEach(p=>p.hidden=p.id!==button.dataset.panel);}));
function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
{{playground}}
</script></body></html>'''
mfu_path=ROOT/'results/mfu.json'
mfu_html='<p class="note">正在补充 profiler 数据。当前不把 GPU Util 或加速倍数冒充 MFU。</p>'
if mfu_path.exists():
    mfu_html=read('results/mfu.json')['report_html']
replace={
'cpu_gain':f"{local['cases'][0]['speedup']:.1f}",'gpu64_gain':f"{gpu['batch_fp64']['gpu_vs_cpu_cached_same_work']:.1f}",'gpu32_gain':f"{gpu['batch_fp32']['gpu_vs_cpu_cached_same_work']:.1f}",
'fixed_loss':f"{quality['runs'][0]['loss']:.1f}",'quality_loss':f"{quality['runs'][2]['loss']:.1f}",'annealed_loss':f"{quality['runs'][1]['loss']:.1f}",'quality_sec':f"{quality['runs'][2]['ms_including_checkpoints']/1000:.2f}",'fit_loss':f"{fit['loss']:.1f}",'fit_sec':f"{fit['gpu_ms']/1000:.2f}",
'fit_drift':f"{fit['max_final_rerender_mse_drift']:.4f}",'single64_gain':f"{gpu['single_fp64']['gpu_vs_cpu_cached_same_work']:.2f}",'fp32_mismatches':str(gpu['single_fp32']['verification']['canonical_fp64_loss_mismatch']),'fp32_error':f"{gpu['single_fp32']['verification']['canonical_fp64_max_mse_error']:.4f}",'fp32_final_mismatches':str(gpu['batch_fp32']['verification']['final_chain_mismatch']),
'fit_memory':f"{fit['workspace_bytes']/1048576:.1f}",'source_url':'https://github.com/zsc/widgets/blob/'+prov['commit']+'/mcmc_splatting.html','commit_short':prov['commit'][:12],'sha256':prov['sha256'],'cpu_rows':cpu_rows,'gpu_rows':gpu_rows,'mfu_html':mfu_html,'core':core,'data':data,'playground':playground}
for k,v in replace.items():page=page.replace('{{'+k+'}}',v)
assert not re.search(r'\{\{[a-z_]+\}\}',page)
(ROOT/'report.html').write_text(page)
print(ROOT/'report.html',len(page.encode()),'bytes')

