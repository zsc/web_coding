#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json,pathlib,base64,re,subprocess,sys
root=pathlib.Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(root/'src/build_report.py')],check=True)
s=json.loads((root/'results/final_summary.json').read_text());r=s['representative'];web=s['webgpu'];old=json.loads((root/'results/desk/zhou_500000.json').read_text())
def img(p):return 'data:image/png;base64,'+base64.b64encode((root/p).read_bytes()).decode()
page=(root/'report.html').read_text()
hero=f'''<section class="hero"><div><div class="eyebrow">LOCAL STUDY · FINAL ACCEPTANCE PASSED · DESK + WEBGPU</div>
<h1>“周舒畅”，<br><em>10 秒内，拟合到笔画。</em></h1><p>从原始单参数 MCMC，走到局部更新与并行 Adam。保留 320×128 的原目标，用完整 FP64 重渲染验收：总耗时连续三次低于 10 秒，MSE 从 95.89 降至约 1.35。</p>
<div class="chips"><span>1,600 GAUSSIANS × 8 CHAINS</span><span>RTX 2080 SUPER / CUDA</span><span>另附可运行 WebGPU / FP16 实验室</span></div></div><div class="hero-art"><strong>{s['process_median_s']:.2f} s / {r['loss']:.2f} MSE</strong><small>END-TO-END PROCESS TIME / FP64 VALIDATED</small></div></section>'''
page=re.sub(r'<section class="hero">.*?</section>',lambda _:hero,page,count=1,flags=re.S)
metrics=f'''<div class="metrics"><div class="metric"><div class="eyebrow">完整进程 · 3次中位数</div><div class="number">{s['process_median_s']:.2f} s</div><p>7.62 / 8.13 / 8.19 秒，全部小于10秒<br>含初始化、计算、校验、文件输出与退出</p></div><div class="metric"><div class="eyebrow">完整 FP64 重渲染</div><div class="number">{r['loss']:.2f}</div><p>三次 MSE：1.340–1.358<br>PSNR {r['psnr']:.2f} dB · 固定 320 × 128 目标</p></div><div class="metric"><div class="eyebrow">网页 FP16 缓存 · 实测</div><div class="number">1.14×</div><p>Apple M5 WebGPU，贡献矩阵内存减半<br>FP32 累加，三轮 GPU timestamp 中位数</p></div></div>'''
page=re.sub(r'<div class="metrics">.*?</div></div>\s*<p class="caption">.*?</p>',lambda _:metrics+'<p class="caption">10 秒验收是预先编译好的拟合进程计时，不含编译、SSH 传输、字体栅格化和报告生成。测试期间保留其他 GPU 进程。新方案用连续未截断强度 MSE 做 Adam 细化，属于优化器变更；最终验收仍使用原版取整、截断后的全图 MSE。</p>',page,count=1,flags=re.S)
rows=''
for name,gs,steps,adam,time,mse in [
 ('上一轮全图 MCMC',800,500000,0,'GPU 77.26 s',95.8914),
 ('局部 MCMC',800,500000,0,'进程内 13.00 s',98.1133),
 ('局部 MCMC',800,1000000,0,'进程内 26.46 s',50.992),
 ('MCMC + Adam',800,10000,1500,'进程内 1.60 s',24.1875),
 ('MCMC + Adam',1200,10000,6000,'完整进程 4.31 s',4.24648),
 ('最终配置 · 3次验收',1600,10000,8000,'完整进程 8.13 s（中位数）',r['loss'])]:
 rows+=f'<tr><td>{name}</td><td>{gs:,}</td><td>{steps:,} + {adam:,}</td><td>{time}</td><td class="green">{mse:.3f}</td></tr>'
webrows=''.join(f'<tr><td>{mode}</td><td>{v["median_gpu_ms"]:.2f} ms</td><td>{v["cache_MiB"]:.0f} MiB</td><td>{v["rerender_loss"]:.3f}</td><td>{v["render_mae"]:.4f}</td></tr>' for mode,v in web.items())
new=f'''<section class="section" id="final-result"><div class="section-head"><div><div class="index">FINAL / “周舒畅” 验收结果</div><h2>笔画变清晰，时间也真正降下来了。</h2></div><p>目标、字体、分辨率保持一致。下面中位耗时对应第二次独立进程运行；文件与原始计时全部保存。</p></div>
<div class="gallery"><div class="image-card"><h3>目标 / 320 × 128</h3><img style="width:100%;margin:20px 0;image-rendering:pixelated" src="{img('assets/zhou_target.png')}"><strong>周舒畅</strong><p>STHeiti Medium<br>4倍栅格化后抗锯齿下采样</p></div><div class="image-card"><h3>旧版 / 50万步 MCMC</h3><img style="width:100%;margin:20px 0;image-rendering:pixelated" src="{img('assets/zhou_500000.png')}"><strong>95.89 MSE</strong><p>800 高斯 × 8 链<br>GPU 77.26 秒（未测完整进程）</p></div><div class="image-card"><h3>最终版 / MCMC + Adam</h3><img style="width:100%;margin:20px 0;image-rendering:pixelated" src="{img('assets/zhou_final.png')}"><strong>{r['loss']:.3f} MSE</strong><p>1,600 高斯 × 8 链<br>完整进程中位数 8.13 秒</p></div><div class="image-card"><h3>绝对像素误差 × 16</h3><img style="width:100%;margin:20px 0;image-rendering:pixelated" src="{img('assets/zhou_error_x16.png')}"><strong>{r['psnr']:.2f} dB</strong><p>亮度放大16倍便于检查<br>没有以裁剪区域代替全图评分</p></div></div>
<div class="panel" style="margin-top:18px"><h3>放大对照：拖动分界查看原图与最终拟合</h3>
<div style="position:relative;max-width:900px;margin:20px auto;background:black;line-height:0">
<img src="{img('assets/zhou_target.png')}" style="display:block;width:100%;image-rendering:pixelated">
<img id="compare-overlay" src="{img('assets/zhou_final.png')}" style="position:absolute;inset:0;width:100%;image-rendering:pixelated;clip-path:inset(0 50% 0 0)">
</div><input id="compare-slider" aria-label="原图与拟合对照分界" type="range" min="0" max="100" value="50" style="width:100%;accent-color:#c4ef89">
<p class="note">分界左侧为拟合，右侧为目标。图像可用本地 assets/zhou_final.png 直接打开。</p></div>
<div class="two" style="margin-top:18px"><div class="panel"><h3>加速来自哪里</h3><ul><li>6σ 椭圆支持区：局部 MCMC 每步只评价新旧支持区并集，更新 ΔSSE，取消 O(CPN) 贡献矩阵。</li><li>warp shuffle 归约减少同步。内存只保留总图、局部差分、参数和 Adam 状态。</li><li>每条链前 10,000 步 MCMC 做局部搜索，后 8,000 步 Adam 同时更新全部高斯；一轮 Adam 有 N 个并行参数更新。</li><li>渲染与梯度都按高斯支持区域执行，优化阶段更能利用 GPU，而非为每个随机提案等待整条链。</li></ul></div><div class="panel"><h3>如何确认质量与计时</h3><ul><li>每次输出选中链的最终参数，独立做<strong>全图、无尾部截断的 FP64 重渲染</strong>，再按原版 round / clip 算 MSE。</li><li>CUDA 的增量 SSE 与实际缓存图像 SSE 相等；最终样本缓存图与完整重渲染像素 MAE 为 {r['pixel_mae']:.6f}。</li><li>六个参数的解析梯度通过中心差分校验，最大相对误差 5.1×10⁻⁹。</li><li>/usr/bin/time 记录整个进程：三次全部通过。Adam 原子加法会带来微小非确定性，MSE 范围完整保留。</li></ul></div></div>
<div class="panel" style="margin-top:18px"><h3>保留所有关键实验，包括未达标方案</h3><div class="table-scroll"><table><thead><tr><th>方案</th><th>高斯</th><th>每链 MCMC + Adam 步数</th><th>计时口径 / 时间</th><th>完整 MSE</th></tr></thead><tbody>{rows}</tbody></table></div><p class="caption">所有中文案例均为 8 链、320×128。表中早期 GPU / 进程内计时不冒充完整进程时间。6σ 忽略尾部单点上界≤255 exp(−18)，1,600颗总上界约0.0062灰度；实际最终误差由完整重渲染验证。新方案用新的目标近似和优化器，不声称与原 MCMC 同轨迹。</p></div>
<div class="panel" style="margin-top:18px"><h3>直接复现最终配置</h3><pre class="code"># 本地复制 / 编译 / 运行 / 拉回结果：
./run_fast.sh
# 已编译后，在 desk 项目目录中：
/usr/bin/time -f 'process_wall_seconds=%e' ./hybrid_fit \\
  --input zhou_target.pgm --gaussians 1600 --chains 8 \\
  --steps 10000 --adam-steps 8000 --output results/fast</pre>
<p class="note">独立脚本 <a href="run_acceptance.sh">run_acceptance.sh</a> 重复三次并另做 profiler 采样。<a href="results/final_summary.json">汇总与源码 SHA-256</a> · <a href="src/roi_fit.cu">最终 CUDA 源码</a> · <a href="assets/zhou_final.png">最终图片</a>。</p></div>
</section>
<section class="section" id="webgpu-results"><div class="section-head"><div><div class="index">WEB / FP16 IS MEASURED, NOT ASSUMED</div><h2>浏览器 GPU 已经跑通，半精度快多少也实测了。</h2></div><p>这台 Apple M5、实际 WebGPU / Metal adapter，shader-f16 与 timestamp-query 可用。网页原型仍是 MCMC；desk 的最终 Adam 管线尚未移植到 WGSL。</p></div><div class="panel">
<div class="table-scroll"><table><thead><tr><th>模式</th><th>GPU 中位数</th><th>贡献矩阵</th><th>完整重渲染 MSE</th><th>缓存图 / 重渲染 MAE</th></tr></thead><tbody>{webrows}</tbody></table></div>
<p class="caption">相同160×64目标、128高斯、4链、每链8,192步；各模式预热、轮换顺序、3次计时。FP16缓存吞吐提升约14.2%，内存减半；进一步把exp改成f16没有明显额外收益。三条轨迹不同，最终MSE略有差异，不代表相同轨迹验证。</p>
<p style="margin-top:20px"><a href="webgpu_lab.html" style="display:inline-block;padding:10px 18px;background:#c4ef89;color:#152310;border-radius:7px;text-decoration:none">打开 WebGPU / FP16 实验室 →</a></p>
<p class="caption">支持320×128、800高斯和更长迭代；这组较大网页配置未作为10秒质量验收。WGSL没有标准BF16标量；此处FP16更适合0–255幅值范围。参数、总和保留FP32，SSE用u32，不能把全部计算强改半精度。下一步网页最有价值的是移植局部区域 / 并行梯度管线；不能把 CUDA 的8.13秒当网页速度。</p></div></section>'''
page=page.replace('<section class="section" id="output">',new+'<section class="section" id="output">',1)
page=page.replace('01 / THE OUTPUT','EARLIER / 小图基础实验').replace('先看真正拟合出来的图。','最初的小图实验与基准保留如下。')
page=page.replace('网页试跑是 JS CPU，远端是 CUDA。当前没有 WebGPU 实现。网页使用 GPU 需要另写 WGSL compute shader 并验证。','本报告下方旧试跑仍为 JS CPU；独立 WebGPU / FP16 实验室已实现并测试。此节保留最初常驻 MCMC 内核设计。')
page=page.replace('ROI 裁剪、快速 exp、Adam、FP32 边界回退和多核 CPU 均需要单独做精度 / 性能消融，当前未声称实现。','局部 ROI 与并行 Adam 已实现并在上方验收。FP32 边界回退、多核 CPU 与更大图像仍未验证。')
mf=f'''<p class="note"><strong class="green">最终渲染 + Adam 梯度两内核采样的 FP32 指令算术利用率约 {s['profile_pair_arithmetic_utilization_pct']:.2f}%</strong>。不是模型训练定义的 MFU，也不是整个程序平均。</p><ul><li>renderSparse：SM throughput 73.23%，occupancy 83.97%。</li><li>refineAdam：SM throughput 66.33%，occupancy 83.07%。</li><li>算术利用率使用实际 FADD + FMUL + 2×FFMA，按两内核时间和实测 SM 时钟加权；不含整数 / SFU / FP64，也不含 memset、MCMC、CPU、传输和空隙。</li></ul><p class="note">先前全图 MCMC 样本为3.85%，SM throughput 9.48%、occupancy 24.98%。新优化器增加跨高斯并行度。两组是不同阶段的内核样本，不能当作同一工作量的整体MFU对比。原始数据见 <a href="results/desk/ncu_final.csv">ncu_final.csv</a>。</p>'''
page=re.sub(r'<div id="mfu-text">.*?</div><p class="note"',lambda _:'<div id="mfu-text">'+mf+'</div><p class="note"',page,count=1,flags=re.S)
page=page.replace('本地：mcmc_splatting_gpu<br>desk：','本地：mcmc_splatting_gpu（最终入口 run_fast.sh）<br>desk：')
page=page.replace('python3 src/build_report.py\nopen report.html','python3 src/summarize_final.py\npython3 src/finalize_report.py\nopen report.html')
page=page.replace('报告中的网页运行后端：CPU JavaScript','主报告旧试跑：CPU / 独立实验室：WebGPU')
page=page.replace('<script type="module">','<script>document.getElementById("compare-slider").addEventListener("input",e=>document.getElementById("compare-overlay").style.clipPath="inset(0 "+(100-Number(e.target.value))+"% 0 0)");</script><script type="module">',1)
from hybrid_report import update_hybrid_report
page=update_hybrid_report(page,root)
from ablation_report import update_ablation_report
page=update_ablation_report(page,root)
from lm_report import update_lm_report
page=update_lm_report(page,root)
from muon_report import update_muon_report
page=update_muon_report(page,root)
from graph_report import update_graph_report
page=update_graph_report(page,root)
(root/'report.html').write_text(page)
print('Final report:',len(page.encode()),'bytes')

