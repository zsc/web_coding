"""Latest Adam optimization report: CUDA diagnostics, tighter ROI, and WebGPU proof."""
import base64,hashlib,json,statistics,struct,zlib

def update_graph_report(page,root):
    read=lambda p:json.loads(p.read_text())
    folder=root/'results/desk_roi_tight_study'
    if not (folder/'study.json').exists():return page
    study=read(folder/'study.json');check=read(folder/'checks.json');graph=read(root/'results/desk_graph_study/study.json');gcheck=read(root/'results/desk_graph_study/checks.json');web=read(root/'results/webgpu_roi_study.json');wcheck=read(root/'results/webgpu_roi_checks.json');profile=read(root/'results/desk_graph/profile_summary.json')
    assert study['complete'] and graph['complete'] and web['complete'] and check['passed'] and gcheck['passed'] and wcheck['passed']
    for data,archive in [(study,'desk_roi_tight_study'),(graph,'desk_graph_study')]:
        for source,digest in data['sources'].items():assert hashlib.sha256((root/'source'/archive/source.split('/')[-1]).read_bytes()).hexdigest()==digest
    for source,digest in web['sources'].items():
        path=root/source if source.startswith('assets/') else root/'source/webgpu_roi'/source.split('/')[-1]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    def stats(runs):
        return {'wall_ms':statistics.median(r.get('wall_ms',r.get('total_ms')) for r in runs),'gpu_ms':statistics.median(r['gpu_ms'] for r in runs),'mse':statistics.median(r['loss'] for r in runs),'wall_range_ms':[min(r.get('wall_ms',r.get('total_ms')) for r in runs),max(r.get('wall_ms',r.get('total_ms')) for r in runs)],'mse_range':[min(r['loss'] for r in runs),max(r['loss'] for r in runs)]}
    names={'original':'原 Adam · 6σ','optimized_6':'FMA + warp + 并行验收 · 6σ','roi_4p5':'推荐 · 4.5σ','roi_4':'更紧裁剪 · 4σ','roi_4p5_long':'更低误差 · 4.5σ / 11,000步'}
    groups={g:stats([r for r in study['runs'] if r['phase']=='repeat' and r['name']==g]) for g in names}
    gnames={'adam_eager':'原 Adam','adam_graph':'只加 Graph（32步/图）','adam_fma':'只开 FMA','adam_parallel':'FMA + 12线程验收','adam_warp':'再改 warp 布局','adam_warp_graph':'优化后再加 Graph'}
    gg={g:stats([r for r in graph['runs'] if r['phase']=='repeat' and r['name']==g]) for g in gnames}
    wnames=['6σ / 单 Worker','4.5σ / 单 Worker','4.5σ / 4 Workers','4σ / 单 Worker'];wg={g:stats([r for r in web['runs'] if r['phase']=='repeat' and r['name']==g]) for g in wnames}
    for g in wg:wg[g]['validation_ms']=statistics.median(r['validation_ms'] for r in web['runs'] if r['phase']=='repeat' and r['name']==g)
    base=groups['original'];fast=groups['roi_4p5'];wb=wg[wnames[0]];wf=wg[wnames[2]]
    held=[]
    for seed in [1042,2042,3042]:
        rows={r['name']:r for r in study['runs'] if r['phase']=='holdout' and r['seed']==seed};held.append({'seed':seed,'mse':{g:r['loss'] for g,r in rows.items()},'wall_ms':{g:r['wall_ms'] for g,r in rows.items()}})
    representatives={g:sorted([r for r in study['runs'] if r['phase']=='repeat' and r['name']==g],key=lambda r:r['loss'])[1]['file'] for g in names}
    summary={'desk':groups,'graph_ablation':gg,'webgpu':wg,'holdout':held,'desk_speedup':base['wall_ms']/fast['wall_ms'],'desk_gpu_speedup':base['gpu_ms']/fast['gpu_ms'],'webgpu_speedup':wb['wall_ms']/wf['wall_ms'],'webgpu_gpu_speedup':wb['gpu_ms']/wf['gpu_ms'],'representatives':representatives,'checks':{'desk':check,'graph_adam':gcheck,'webgpu':wcheck},'timing_note':'Each study has its own contemporaneous baseline; no speedup uses different machines or historical uncontrolled runs.'}
    (root/'results/adam_acceleration_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    def png(pixels,path):
        def chunk(k,d):return struct.pack('>I',len(d))+k+d+struct.pack('>I',zlib.crc32(k+d)&0xffffffff)
        raw=b''.join(b'\0'+pixels[y*320:(y+1)*320] for y in range(128));path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',320,128,8,0,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))
    def embedded(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
    target=bytes(read(root/'assets/zhou_target.json')['pixels']);cards=''
    for g in ['original','roi_4p5','roi_4p5_long']:
        r=read(folder/representatives[g]);raw=(folder/representatives[g].replace('.json','.pgm')).read_bytes();header=b'P5\n320 128\n255\n';assert raw.startswith(header);pixels=raw[len(header):]
        fit=root/f'assets/adam_accel_{g}.png';error=root/f'assets/adam_accel_{g}_error.png';png(pixels,fit);png(bytes(min(255,abs(a-b)*16) for a,b in zip(pixels,target)),error)
        cards+=f'<div class="image-card"><h3>{names[g]}</h3><img alt="{names[g]}拟合" src="{embedded(fit)}" style="width:100%;margin:16px 0"><strong>{r["loss"]:.4f} MSE</strong><p>绝对像素误差 × 16</p><img alt="{names[g]}误差" src="{embedded(error)}" style="width:100%"></div>'
    rows=''.join(f'<tr><td>{names[g]}</td><td>{r["wall_ms"]/1000:.3f} s</td><td>{r["gpu_ms"]/1000:.3f} s</td><td>{r["mse"]:.4f}</td></tr>' for g,r in groups.items())
    graphrows=''.join(f'<tr><td>{gnames[g]}</td><td>{r["wall_ms"]/1000:.3f} s</td><td>{r["gpu_ms"]/1000:.3f} s</td><td>{r["mse"]:.4f}</td></tr>' for g,r in gg.items())
    webrows=''.join(f'<tr><td>{g}</td><td>{r["wall_ms"]/1000:.3f} s</td><td>{r["gpu_ms"]/1000:.3f} s</td><td>{r["validation_ms"]/1000:.3f} s</td><td>{r["mse"]:.4f}</td></tr>' for g,r in wg.items())
    heldrows=''.join(f'<tr><td>{r["seed"]}</td><td>{r["mse"]["original"]:.4f}</td><td>{r["mse"]["roi_4p5"]:.4f}</td><td>{r["mse"]["roi_4"]:.4f}</td><td>{r["mse"]["roi_4p5_long"]:.4f}</td></tr>' for r in held)
    pilot=read(root/'results/desk_roi_tight_pilot/pilot.json');prows=''.join(f'<tr><td>{r["roi_sigma"]:g}σ</td><td>{r["wall_ms"]/1000:.3f} s</td><td>{r["loss"]:.4f}</td><td>{r["max_sum_drift"]:.5f}</td></tr>' for r in pilot['runs'])
    section=f'''<section class="section" id="desk-graph-results"><div class="section-head"><div><div class="index">NEW / ADAM · NSIGHT → CUDA → WEBGPU</div><h2>Adam 继续降到约 2 秒，<br>网页也实测到了收益。</h2></div><p>保持“周舒畅”320×128、1,600高斯 × 8链。推荐配置仍为8,000步；先用 Nsight 找瓶颈，再分别验证 FMA、线程布局、验收并行与 ROI 裁剪。</p></div>
<div class="panel"><h3>desk · RTX 2080 SUPER · 三轮中位数</h3><div class="table-scroll"><table style="max-width:850px"><thead><tr><th>方案</th><th>完整进程</th><th>GPU 计算</th><th>全图 MSE ↓</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="caption">推荐方案相对本轮原 Adam：完整耗时快 {base['wall_ms']/fast['wall_ms']:.2f}×，减少 {100*(1-fast['wall_ms']/base['wall_ms']):.1f}%；GPU 计算快 {base['gpu_ms']/fast['gpu_ms']:.2f}×。MSE 变化 {100*(fast['mse']/base['mse']-1):+.2f}%。包含初始化、GPU、完整 FP64 验收、输出与退出；不含编译、SSH。这里使用本轮基线，不与不同阶段的耗时混算。</p></div>
<div class="gallery" style="margin-top:18px">{cards}</div><p class="caption">图片取各组 MSE 居中的一次。11,000步是质量选项，不标作等时实验。此前时间校准的11,840步实际超出目标预算，原始结果保留在 Graph 实验记录中，不用于“同耗时”结论。</p>
<div class="panel" style="margin-top:18px"><h3>WebGPU · 本机 Apple M5 · 三轮中位数</h3><div class="table-scroll"><table style="max-width:850px"><thead><tr><th>方案</th><th>总耗时</th><th>GPU</th><th>Float64 验收</th><th>全图 MSE ↓</th></tr></thead><tbody>{webrows}</tbody></table></div>
<p class="caption">4.5σ + 4 Workers 相对同轮6σ基线快 {wb['wall_ms']/wf['wall_ms']:.2f}×，耗时减少 {100*(1-wf['wall_ms']/wb['wall_ms']):.1f}%，MSE 变化 {100*(wf['mse']/wb['mse']-1):+.2f}%。每次新建管线，包含编译、初始化、计算、回传、Worker 启动与全图验收；先做一次预热，再轮换3轮。不含页面下载和计时外独立串行交叉检查。浏览器/驱动可能复用编译缓存。</p>
<p><a href="adam_roi_lab.html" style="display:inline-block;background:#c4ef89;color:#152310;padding:12px 18px;border-radius:7px;text-decoration:none">直接运行加速版 Adam / 重跑对照 →</a></p>
<p>这次网页迁移的是更紧的 ROI 和多 Worker 验收。原网页已经使用32线程工作组、subgroup reduction、每批128步提交；不能把这些再次计作新增收益。原网页验收已在单 Worker 中执行，这次改为四个不重叠行分区，每个像素仍累加全部高斯。</p></div>
<details class="panel" style="margin-top:18px"><summary>CUDA Graph 为什么没有明显加速？Nsight 测到了什么？</summary>
<div class="table-scroll"><table style="max-width:850px"><thead><tr><th>第一阶段消融 · 均为6σ</th><th>完整进程</th><th>GPU</th><th>MSE</th></tr></thead><tbody>{graphrows}</tbody></table></div>
<p>实际实现了 CUDA Graph，每次捕获32步并重放；迭代计数在 GPU 上递增，余弦学习率与动量偏差修正连续推进。另测了1、128、512步/图。建图、实例化与上传均计入完整耗时；重放 GPU 时间单独报告。</p>
<p>Nsight Systems 的原 Adam 时间线中，kernel / memset 区间覆盖约 {profile['nsys']['adam_eager']['covered_interval_pct']:.1f}%。这不是 SM occupancy，也不是 MFU；API 调用时间包含队列反压，不能把它全部算成启动开销。Graph32 把优化循环提交降为250次，但实测总耗时基本不变。</p>
<p>Nsight Compute 指出乘加未融合、线程执行不均匀等问题。开启 --fmad=true 后保留 FP32；每个 warp 处理一颗高斯减少部分梯度开销。局部采样中，梯度 kernel 从约147µs降到125µs，而渲染从约67µs变为71µs；应看管线净收益，不能把某个 kernel 的局部变化当整体加速。12线程验收把第一阶段的约1.07秒降到约0.14秒。</p>
<p><a href="results/desk_graph/profile_summary.json">Nsight 汇总</a> · <a href="results/desk_graph/adam_eager.nsys-rep">原始时间线</a> · <a href="results/desk_graph/adam_graph32.nsys-rep">Graph 时间线</a> · <a href="results/desk_graph/adam_ncu.csv">原 kernel 指标</a> · <a href="results/desk_graph/optimized_ncu.csv">优化 kernel 指标</a>。Profiler 数据只用于诊断，正式速度来自未挂 profiler 的独立进程。</p>
<p>WebGPU 的 <a href="https://www.w3.org/TR/webgpu/#dom-gpuqueue-submit">GPUCommandBuffer 只能提交一次</a>，没有直接对应本次 CUDA Graph 重放的接口。可继续批量编码与减少读回。WGSL 有 <a href="https://www.w3.org/TR/WGSL/#fma-builtin">fma</a>，但 CUDA 编译开关及收益不能照搬到 Metal；此次网页没有把显式 FMA 当成已测收益。</p></details>
<details class="panel" style="margin-top:18px"><summary>裁剪误差、未参与调参的种子和完整验收</summary>
<p>ROI 从6σ收至4.5σ，包围框面积按几何比例约减少44%，但实际速度还受固定开销、形状及硬件影响。计算阶段省略很小的尾部；最终评分始终用未裁剪、未量化近似的完整 FP64 / Float64 渲染。缓存 MSE 不代替验收分数。4σ更快，截断误差也更大；3.5σ在首轮明显降低质量，因此未选为推荐方案。</p>
<div class="table-scroll"><table style="max-width:850px"><thead><tr><th>首轮试探</th><th>完整耗时</th><th>全图 MSE</th><th>最大浮点强度差</th></tr></thead><tbody>{prows}</tbody></table></div>
<h3>固定配置后的新初值 · 全图 MSE</h3><div class="table-scroll"><table style="max-width:850px"><thead><tr><th>seed基数</th><th>原 Adam</th><th>4.5σ</th><th>4σ</th><th>4.5σ / 11k步</th></tr></thead><tbody>{heldrows}</tbody></table></div>
<p>每组仍是8条链，seed+1009×链号；未用这些种子选择配置。4.5σ在新种子的完整耗时为 {min(r['wall_ms']['roi_4p5'] for r in held)/1000:.3f}–{max(r['wall_ms']['roi_4p5'] for r in held)/1000:.3f} 秒。所有GPU其他负载保留，没有锁频或清缓存。此结果只覆盖本目标和规模，更大高斯重叠密度需重新检查尾部误差。</p>
<p>Graph阶段 {gcheck['checked']} 份 Adam 输出、ROI阶段 {check['checked']} 份输出，经原始 JavaScript 渲染器逐像素重算全部一致；Graph非整批尾部的计数与短程参数对照通过。网页13次运行均做完整 Float64 验收，4组代表输出另经原版渲染器复核，四路 Worker 与串行输出逐像素一致。</p>
<p><a href="results/desk_graph_study/checks.json">Graph / Adam校验</a> · <a href="results/desk_roi_tight_study/checks.json">ROI校验</a> · <a href="results/webgpu_roi_checks.json">网页校验</a> · <a href="results/adam_acceleration_summary.json">完整汇总</a> · <a href="results/webgpu_roi_study.json">网页原始结果</a>。</p></details>
<details class="panel" style="margin-top:18px"><summary>复现与代码</summary><pre>./run_adam_accelerated.sh
# desk 上等价执行：
./adam_roi_4p5 --mode 0 --warp-roi 1 --verify-threads 12 --steps 8000 --output results/adam_fast
# Graph 实验开关：--graph-batch 32（默认0，实测无明显收益）</pre>
<p><a href="src/graph_fit.cu">CUDA实现</a> · <a href="src/parallel_validate.mjs">网页多Worker验收</a> · <a href="src/hybrid_webgpu.mjs">WGSL实现</a> · <a href="source/desk_roi_tight_study/graph_fit.cu">实测CUDA快照</a>。推荐入口仍使用 FP32，没有把新的收益归到 FP16 / BF16。旧报告中的 FP16 实验独立保留。</p></details></section>'''
    page=page.replace('<section class="section" id="desk-muon-results">',section+'<section class="section" id="desk-muon-results">',1)
    page=page.replace('<a href="#desk-muon-results">','<a href="#desk-graph-results">Adam加速</a><a href="#desk-muon-results">',1)
    return page
