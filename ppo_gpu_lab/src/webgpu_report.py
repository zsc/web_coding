import json, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/'results'/name).read_text())
def render():
 b=read('webgpu_benchmark.json');v=read('webgpu_validation_worker.json');a=read('webgpu_independent_eval.json')
 r=read('webgpu_train_seed71.json');runs=[read(f'webgpu_train_seed{s}.json') for s in [71,79,83]]
 selected=a['runs'][0]['result']
 def count(x):return f"{x['success']} / {x['count']} ({x['successRate']*100:.1f}%)"
 quality=''.join(f"<tr><td>{label}</td><td>{count(a['original10N'][key])}</td><td>{count(a['initial40N'][key])}</td><td>{count(selected[key])}</td></tr>" for label,key in [('±2° 静止释放','angle2'),('±5° 静止释放','angle5'),('±10° 静止释放','angle10'),('±2° + 三次 ±0.35 m/s 冲击','pushes')])
 seedrows=''.join(f"<tr><td>{ar['trainingSeed']}</td><td>{ar['checkpoint']} / 1200</td><td>{count(ar['result']['angle10'])}</td><td>{count(ar['result']['pushes'])}</td><td>{run['training']['trainingMs']/1000:.2f} s / {run['training']['activeMs']/1000:.2f} s</td></tr>" for ar,run in zip(a['runs'],runs))
 checks=''.join(f'<tr><td>{k}</td><td>{value:.3e}</td><td>&lt; {v["thresholds"][k]:.0e}</td></tr>' for k,value in v['checks'].items())
 colors=['#15795b','#b45a34','#627c9c'];lines=''
 for run,color in zip(runs,colors):
  points=' '.join(f'{45+h["iteration"]/1200*610:.1f},{195-h["evaluation"]["successRate"]*165:.1f}' for h in run['history'])
  lines+=f'<polyline points="{points}" stroke="{color}" fill="none" stroke-width="2.5"/>'
 chart=f'<svg class="chart" viewBox="0 0 680 235" role="img" aria-label="三个训练种子的验证恢复率曲线"><path d="M45 30H655M45 112.5H655M45 195H655" stroke="#ccd5cb"/><text x="8" y="34">100%</text><text x="14" y="116">50%</text><text x="20" y="199">0%</text><text x="45" y="222">0 轮</text><text x="335" y="222">600</text><text x="615" y="222">1200 轮</text>{lines}</svg>'
 return f'''<section id="webgpu-results" style="padding-top:42px">
 <div class="section-title"><span class="number">WEBGPU</span><h2>在网页里训练，松手就能测试。</h2><span class="aside">APPLE M5 · 10 GPU CORES · CHROME 155</span></div>
 <p>完整的采样、物理仿真、GAE、PPO 反向传播和 Adam 已迁到 WGSL。训练在 Web Worker 中运行，页面可以开始、暂停、继续，查看进度与曲线；结束后自动将最佳验证快照接入拖拽实验。下方已经载入固定 seed 71 的实测模型。</p>
 <p><a href="webgpu.html" style="display:inline-block;padding:12px 22px;background:#15795b;color:#fff;border-radius:5px;text-decoration:none">打开 WebGPU 训练工作台 ↗</a>　<a href="#webgpu-method">看测量与验证方法 ↓</a></p>
 <div class="metrics"><div class="metric"><strong>{b['medianWallMs']:.2f} ms</strong><span>一轮完整训练中位数</span></div><div class="metric"><strong>{b['sps']/10000:.1f} 万</strong><span>环境步 / 秒，含 80 次参数更新</span></div><div class="metric"><strong>{r['training']['trainingMs']/1000:.2f} s</strong><span>seed 71：1,200 轮累计训练时间</span></div><div class="metric"><strong>{selected['angle10']['successRate']*100:.1f}%</strong><span>新测试集：±10° 姿态恢复</span></div></div>
 <div class="two"><div class="panel"><h3>继续训练后，恢复范围扩大</h3>{chart}<div class="legend"><span style="color:{colors[0]}">● seed 71</span><span style="color:{colors[1]}">● seed 79</span><span style="color:{colors[2]}">● seed 83</span></div><p class="small">每 50 轮在固定 128 组 ±10° 状态上验证。以恢复率选快照，同分看回报。曲线不是最终测试集。</p></div>
 <div class="reading"><p><b>先学会小扰动，再扩到 ±10°。</b>从已有 CUDA PPO seed 7 权重继续训练，前 200 轮扩大重置角度。60% 重置用宽分布，40% 保留近竖直状态；加入位置、速度及角速度变化，每步 0.3% 概率给小车速度冲击。</p><p><b>网络直接控制小车。</b>保留 819 个 FP32 参数，使用 40 N 力上限、60° 终止角和更强角度惩罚。训练、评估、拖拽测试均没有 LQR 教师或辅助控制。旧 PPO 在 10 N 学会的权重直接切到 40 N 会失稳，因此同时列出旧 10 N 与同力上限的训练前基线。</p><p>默认 1,200 轮共 19,660,800 个新样本。界面暂停保留环境、参数、随机数和 Adam 状态。可导出 819 个参数、策略、曲线与每轮计时；页面关闭后训练停止。</p></div></div>
 <h3>冻结配置后的新测试集</h3><div class="table-wrap"><table><thead><tr><th>测试条件</th><th>旧 PPO · 10 N</th><th>训练前权重 · 40 N</th><th>WebGPU PPO · 40 N / seed 71</th></tr></thead><tbody>{quality}</tbody></table></div>
 <p class="small">每档 1,024 个状态，测试 seed 271828；在确定训练配置和快照后才运行，未参与选模。初始 x ∈ ±0.25 m，两杆角度独立均匀采样，速度为零；冲击发生在 1、3、5 秒。成功要求 10 秒未越界，最后连续 1 秒满足两角 &lt;2°、角速度 &lt;0.15 rad/s、小车速度 &lt;0.15 m/s。使用独立于 WGSL 实现的 JavaScript Float64 模拟器，尚未对照外部物理引擎。</p>
 <div class="table-wrap"><table><thead><tr><th>训练 seed</th><th>选中轮 / 总轮数</th><th>±10° 恢复</th><th>三次速度冲击</th><th>训练 / 含评估与读回的活跃时间</th></tr></thead><tbody>{seedrows}</tbody></table></div>
 <p class="notice">三个种子都从相同旧权重继续训练；不是三次从随机网络开始。早期探索曾比较 600 与 1,200 轮，随后冻结 1,200 轮及训练设置；上述新测试集用于最终核验。固定使用 seed 71 展示，未按新测试集挑选种子。强拖拽仍可能失败，当前没有从倒下姿态起摆的训练。</p>
 <details id="webgpu-method" open><summary>速度实测：计时包含什么？</summary>
 <p>本机 Apple M5（10 核 GPU），浏览器暴露 apple / metal-3，Chrome 155。固定原任务：修正后的物理、10 N、12° 终止角、无随机冲击；256×64 采样、minibatch 2048、10 epochs，每轮 80 次 Adam 更新。5 轮预热后测 30 轮；中位数 {b['medianWallMs']:.2f} ms，P10–P90 为 {b['p10WallMs']:.2f}–{b['p90WallMs']:.2f} ms，GPU timestamp 内核中位数 {b['medianGpuMs']:.2f} ms。</p>
 <p>完整训练耗时包含 CPU Fisher–Yates 洗牌、buffer 上传、命令编码、提交与 GPU 完成同步。GPU 内部完成 rollout / GAE / 标准化 / 前后向 / 梯度归约 / Adam。吞吐排除编译、评估、界面绘图及 timestamp 读回；表中的活跃时间额外包含评估与读回。训练在 Worker 内，主线程持续显示互动模拟。</p>
 <p>分配的 WebGPU buffer 合计 {b['device']['allocatedBytes']/1048576:.2f} MiB，含 staging；不是浏览器进程总内存。首次开发运行观测 setup 0.24 s，缓存热后只有数毫秒；没有清空驱动 / shader 缓存做冷启动测量。后台负载、浏览器 JIT 和频率变化会影响速度，训练三次的耗时分别如上。</p>
 <p><b>与 desk 的 17.03 ms 不构成纯软件加速倍率。</b>desk 是 RTX 2080 SUPER，WebGPU 用本机 Apple M5；机器、API、编译器和计时边界均不同。保留旧 CUDA 数据供参考，不据此宣称 WebGPU 比 CUDA 快若干倍。</p></details>
 <details><summary>正确性：先对照 PyTorch，再看学习结果</summary><div class="table-wrap"><table><thead><tr><th>项目</th><th>最大误差 / 指标</th><th>容差</th></tr></thead><tbody>{checks}</tbody></table></div><p>2048 样本的相同 PPO minibatch 对照 PyTorch CPU FP32 autograd，覆盖 clip 分支，并比较连续 20 次 Adam。另检查 2048 组物理单步、16384 样本 GAE（真终止与超时截断）。{v['rolloutTransitions']} 条非终止 transition 检查价值连续性，含 {v['observedImpulses']} 次观察到的冲击及跨 rollout 边界，误差为零。数值容差通过不意味着长轨迹逐位一致。</p></details>
 <details><summary>实现取舍与本地复现</summary><p>采样一条 invocation 负责一个环境的 64 步；每个样本融合前向与反向，按参数做 workgroup reduction，避免浮点原子；连续 buffer 保存权重、激活和 Adam。CPU 只准备洗牌索引与调度，完整训练没有 CPU 梯度回退。使用 WebGPU 核心 FP32 功能，timestamp-query 为可选功能。</p><pre>cd /Users/zsc/Downloads/web_coding/ppo_gpu_lab
python3 -m http.server 8778 --bind 127.0.0.1
# 浏览器打开 http://127.0.0.1:8778/webgpu.html
# 连接 GPU → 运行数值校验 / 测量训练速度 → 开始训练
# 独立审计：node src/evaluate_webgpu.mjs
# 重建：python3 src/build_webgpu.py && python3 src/build_report.py</pre><p>报告与已训练模型可离线浏览；重新训练推荐 localhost / HTTPS 和支持 WebGPU 的浏览器。页面不依赖 CDN，不上传训练数据。训练权重已内嵌，不支持 WebGPU 时也可以拖拽测试。</p></details>
 <div class="files"><a href="results/webgpu_benchmark.json">30 轮原始速度数据 ↗</a><a href="results/webgpu_validation_worker.json">数值验证与 rollout 边界结果 ↗</a><a href="results/webgpu_independent_eval.json">新测试集 / 三种子完整结果 ↗</a><a href="results/webgpu_train_seed71.json">seed 71 权重、曲线与每轮计时 ↗</a><a href="src/webgpu_ppo.mjs">WGSL 训练内核与 WebGPU 调度 ↗</a><a href="src/webgpu_worker.mjs">Worker 训练与暂停控制 ↗</a></div>
 </section>'''
