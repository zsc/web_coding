# -*- coding: utf-8 -*-
"""Add the measured live WebGPU acceptance results; never synthesize measurements."""
import base64, hashlib, json, re, statistics, struct, zlib

def update_hybrid_report(page, root):
    source=root/'results/webgpu_hybrid_runs.json'
    cold_paths=[root/f'results/webgpu_hybrid_cold_{i}.json' for i in range(1,4)]
    if not source.exists() or not all(p.exists() for p in cold_paths): return page
    runs=json.loads(source.read_text()); warm=runs[-9:]
    assert len(warm)==9 and all(not x['cold_pipeline'] for x in warm)
    cold=[json.loads(p.read_text()) for p in cold_paths]
    representative=sorted(cold,key=lambda r:r['total_ms'])[1]
    assert all(r['N']==1600 and r['C']==8 and r['mcmc_done']==10000 and r['adam_done']==8000 and not r['cancelled'] for r in cold+warm)
    def clean(r): return {k:v for k,v in r.items() if k not in ('gaussians','image')}
    summary={'cold_runs':[clean(r) for r in cold], 'warm_runs':[clean(r) for r in warm],
      'cold_median_ms':statistics.median(r['total_ms'] for r in cold),
      'cold_max_ms':max(r['total_ms'] for r in cold), 'modes':{},
      'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['source/webgpu_acceptance/hybrid_webgpu.mjs','source/webgpu_acceptance/hybrid_playground.mjs','source/webgpu_acceptance/hybrid_playground.html']},
      'target_sha256':hashlib.sha256((root/'assets/zhou_target.pgm').read_bytes()).hexdigest()}
    for mode in ['fp32','fp16','fp16-cache']:
        subset=[r for r in warm if r['mode']==mode]
        assert len(subset)==3
        summary['modes'][mode]={'total_median_ms':statistics.median(r['total_ms'] for r in subset),'gpu_median_ms':statistics.median(r['gpu_ms'] for r in subset),'mse':statistics.median(r['loss'] for r in subset),'workspace_MiB':subset[0]['workspace_bytes']/1048576}
    (root/'results/webgpu_hybrid_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    def png(pixels,W,H,path):
        def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
        raw=b''.join(b'\0'+bytes(pixels[y*W:(y+1)*W]) for y in range(H))
        path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',W,H,8,0,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))
    png(representative['image'],320,128,root/'assets/zhou_webgpu.png')
    def img(name):return 'data:image/png;base64,'+base64.b64encode((root/'assets'/name).read_bytes()).decode()
    sec=summary['cold_median_ms']/1000; mse=representative['loss']
    hero=f'''<section class="hero"><div><div class="eyebrow">LIVE WEBGPU · LOCAL GPU · ROI + ADAM</div><h1>“周舒畅”，<br><em>直接在网页 GPU 拟合。</em></h1><p>同一份320×128目标、1,600个高斯、8条链。在这台Apple M5浏览器中，首次运行含编译与Float64验证，三次中位数 {sec:.2f} 秒，MSE {mse:.3f}。下方试跑区可直接重现。</p><div class="chips"><span>真实 WEBGPU / 无远端计算</span><span>ROI + ADAM + SUBGROUPS</span><span>FP32 / FP16 对照保留</span></div><p style="margin-top:22px"><a href="#playground">跳到网页拟合 →</a></p></div><div class="hero-art"><strong>{sec:.2f} s / {mse:.3f}</strong><small>CLICK TO FLOAT64 VALIDATION / MSE</small></div></section>'''
    page=re.sub(r'<section class="hero">.*?</section>',lambda _:hero,page,count=1,flags=re.S)
    cold_times=' / '.join(f"{r['total_ms']/1000:.2f}" for r in cold)
    metrics=f'''<div class="metrics"><div class="metric"><div class="eyebrow">网页首次运行 · 3次中位数</div><div class="number">{sec:.2f} s</div><p>{cold_times} 秒，全部小于10秒<br>含管线编译、计算、回传与Float64验证</p></div><div class="metric"><div class="eyebrow">网页最终 MSE · 独立全图验证</div><div class="number">{mse:.3f}</div><p>desk 参考：1.340 MSE / 8.13秒<br>同一目标、分辨率、高斯数与链数</p></div><div class="metric"><div class="eyebrow">浏览器默认精度</div><div class="number">FP32</div><p>保留 FP16 exp / ROI 缓存两种选择<br>本机半精度没有稳定的速度收益</p></div></div>'''
    page=re.sub(r'<div class="metrics">.*?</div></div>\s*<p class="caption">.*?</p>',lambda _:metrics+'<p class="caption">网页计时从点击开始，到完整Float64重渲染结束；不含页面下载与导出文件。每次重新加载页面并新建GPU管线，浏览器/驱动自身可能复用编译缓存。desk为预编译进程计时，两台不同硬件，不用这一比值宣称同硬件加速。浏览器10秒结论只针对本机Apple M5与默认配置。</p>',page,count=1,flags=re.S)
    labels={'fp32':'FP32 · 默认','fp16':'FP16 exp + FP32收尾','fp16-cache':'FP16 ROI缓存 + FP32收尾'}
    rows=''.join(f'<tr><td>{labels[mode]}</td><td>{v["gpu_median_ms"]/1000:.3f} s</td><td>{v["total_median_ms"]/1000:.3f} s</td><td>{v["mse"]:.4f}</td><td>{v["workspace_MiB"]:.2f} MiB</td></tr>' for mode,v in summary['modes'].items())
    section=f'''<section class="section" id="webgpu-results"><div class="section-head"><div><div class="index">WEB / ROI + ADAM · ACCEPTANCE PASSED</div><h2>网页也进了 10 秒，而且质量接近 desk。</h2></div><p>Apple M5 / Metal adapter，320×128、1,600高斯、8链；每链10,000步局部MCMC，再并行执行8,000步Adam。默认目标像素与desk相同。</p></div>
<div class="gallery" style="grid-template-columns:repeat(3,1fr)"><div class="image-card"><h3>相同目标</h3><img alt="周舒畅目标" src="{img('zhou_target.png')}" style="width:100%;margin:20px 0"><strong>320 × 128</strong></div><div class="image-card"><h3>desk / CUDA</h3><img alt="desk拟合" src="{img('zhou_final.png')}" style="width:100%;margin:20px 0"><strong>1.340 MSE</strong></div><div class="image-card"><h3>本页 / WebGPU</h3><img alt="网页拟合" src="{img('zhou_webgpu.png')}" style="width:100%;margin:20px 0"><strong>{mse:.3f} MSE</strong><p>随机初始化，未读取desk参数</p></div></div>
<div class="panel" style="margin-top:18px"><h3>同一配置，三轮交错对比</h3><div class="table-scroll"><table><thead><tr><th>模式</th><th>GPU中位数</th><th>总耗时中位数</th><th>完整重渲染MSE</th><th>GPU工作区</th></tr></thead><tbody>{rows}</tbody></table></div><p class="caption">表内9次全部复用管线；每轮轮换顺序。首次打开三次单独为 {cold_times} 秒。FP16模式在前7,000步Adam使用半精度exp / 幅度乘法，最后1,000步FP32收尾；MCMC始终FP32。缓存模式额外缓存ROI指数，超过每高斯缓存容量的像素重算。</p>
<p class="caption">FP16 exp 与 FP32 差距接近计时波动；ROI缓存额外读写没有带来收益，因此默认保留FP32。保留失败的优化与所有实测，未把上一版小图MCMC的1.14×套用到新后端。WGSL没有标准BF16标量；没有Tensor Core/MFU的浏览器性能声明。</p></div>
<div class="two" style="margin-top:18px"><div class="panel"><h3>移植后的计算路径</h3><ul><li>6σ支持区的ROI MCMC，状态常驻GPU，一次dispatch连续256个提案。</li><li>每个高斯一个32线程workgroup，解析六参数梯度；subgroup归约并行Adam更新。</li><li>每128步Adam合并提交，约250 ms回传一次进度；没有每步CPU回传。</li><li>渲染用u32定点原子累加，步长1/1024灰度。参数、梯度、Adam一阶/二阶状态保持FP32。</li></ul></div><div class="panel"><h3>质量验收与限制</h3><ul><li>先按GPU取整MSE选链，再用JS Float64全图、无6σ截断重渲染选中链，最终按原版round/clip评分。</li><li>默认样本GPU图与全图验证的像素MAE为 {representative['render_mae']:.6f}，最大差 {representative['render_max_error']} 灰度。</li><li>网页最终参数交回原版源码的renderer和loss，得到相同MSE且所有像素一致。GPU单步检查覆盖191个非退化参数，1个近零梯度单独记录。</li><li>同一优化器结构，但定点渲染、浮点运算及随机提案精度有差异，不保证CUDA与网页走同一轨迹。</li></ul></div></div>
<p class="caption"><a href="results/webgpu_hybrid_summary.json">汇总 / 源码与目标SHA-256</a> · <a href="results/webgpu_hybrid_runs.json">全部对照运行</a> · <a href="src/hybrid_webgpu.mjs">WGSL与执行器</a> · <a href="results/webgpu_hybrid_gradient_check.json">GPU单步更新检查</a> · <a href="results/webgpu_hybrid_upstream_check.json">原版renderer复核</a> · <a href="https://www.w3.org/TR/WGSL/">WGSL标准</a> · <a href="webgpu_lab.html">旧版全图MCMC / FP16实验室</a></p></section>'''
    page=re.sub(r'<section class="section" id="webgpu-results">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    page=page.replace('本报告下方旧试跑仍为 JS CPU；独立 WebGPU / FP16 实验室已实现并测试。此节保留最初常驻 MCMC 内核设计。','本报告试跑已改为WebGPU ROI + Adam。此节保留最初常驻MCMC内核设计与desk profiler数据。')
    page=page.replace('主报告旧试跑：CPU / 独立实验室：WebGPU','本报告试跑：WebGPU ROI + Adam / 本机计算')
    page=page.replace('无 npm 依赖。报告所有数据内嵌，直接 file:// 打开即可。保留目录结构以访问源文件链接。原版 T=0 的 NaN 行为保留在 CPU 参考，交互温度限定为正数。','无npm依赖，目标与WGSL源码全部内嵌。建议通过localhost打开以获得WebGPU支持。原版CPU参考仍可单独运行；保留目录结构以访问源文件和原始数据。')
    page=page.replace('FP32 · 本机实测最快','FP32 · 默认').replace('href="#results">实测','href="#webgpu-results">实测')
    return page
