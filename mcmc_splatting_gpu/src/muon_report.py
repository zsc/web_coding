"""Render the Muon experiment, keeping measured data and tuning history visible."""
import base64,hashlib,json,statistics,struct,zlib

def update_muon_report(page,root):
    folder=root/'results/desk_muon_study'
    if not (folder/'study.json').exists():return page
    read=lambda path:json.loads(path.read_text())
    study=read(folder/'study.json');check=read(folder/'renderer_check.json');audit=read(root/'results/desk_muon_pilot/update_check.json')
    pilot=read(root/'results/desk_muon_pilot/pilot.json');refine=read(root/'results/desk_muon_refine/refine.json')
    assert study['complete'] and check['passed'] and audit['passed']
    for data,archive in [(pilot,'desk_muon_pilot'),(refine,'desk_muon_refine'),(study,'desk_muon_study')]:
        for source,digest in data['sources'].items():
            assert hashlib.sha256((root/'source'/archive/source.split('/')[-1]).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((root/'assets/zhou_target.pgm').read_bytes()).hexdigest()==study['target_sha256']
    names={'adam_default':'原参数 Adam','adam_tuned':'调参后 Adam','muon_joint':'Muon · N×6','muon_grouped':'Muon · 两个 N×2 + Adam'}
    groups={}
    for name in names:
        rows=[r for r in study['runs'] if r['phase']=='repeat' and r['group']==name];assert len(rows)==3
        groups[name]={'steps':rows[0]['steps'],'lr':rows[0]['lr'],'floor':rows[0]['floor'],
            'wall_median_ms':statistics.median(r['wall_ms'] for r in rows),
            'wall_range_ms':[min(r['wall_ms'] for r in rows),max(r['wall_ms'] for r in rows)],
            'gpu_median_ms':statistics.median(r['gpu_ms'] for r in rows),
            'mse_median':statistics.median(r['loss'] for r in rows),
            'mse_range':[min(r['loss'] for r in rows),max(r['loss'] for r in rows)]}
    holdout=[]
    for seed in study['holdout_seed_bases']:
        rows={r['group']:r for r in study['runs'] if r['phase']=='holdout' and r['seed']==seed};assert len(rows)==4
        holdout.append({'seed':seed,'mse':{g:r['loss'] for g,r in rows.items()},'wall_ms':{g:r['wall_ms'] for g,r in rows.items()}})
    baseline=groups['adam_default']['wall_median_ms']
    time_delta={g:100*(r['wall_median_ms']/baseline-1) for g,r in groups.items()}
    reps={g:sorted([r for r in study['runs'] if r['phase']=='repeat' and r['group']==g],key=lambda r:r['loss'])[1]['file'] for g in names}
    best_muon=min(groups[g]['mse_median'] for g in ['muon_joint','muon_grouped'])
    adam_wins=groups['adam_tuned']['mse_median']<best_muon
    summary={'groups':groups,'holdout':holdout,'time_difference_percent':time_delta,'representatives':reps,'update_check':audit,'renderer_check':check,'study':study,'adam_beats_tested_muon':adam_wins}
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    def png(pixels,path):
        def chunk(k,d):return struct.pack('>I',len(d))+k+d+struct.pack('>I',zlib.crc32(k+d)&0xffffffff)
        raw=b''.join(b'\0'+pixels[y*320:(y+1)*320] for y in range(128))
        path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',320,128,8,0,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))
    embedded=lambda p:'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
    target=bytes(read(root/'assets/zhou_target.json')['pixels']);cards=''
    for g in ['adam_tuned','muon_joint','muon_grouped']:
        result=read(folder/reps[g]);raw=(folder/reps[g].replace('.json','.pgm')).read_bytes();header=b'P5\n320 128\n255\n';assert raw.startswith(header);pixels=raw[len(header):]
        fit=root/f'assets/desk_muon_{g}.png';error=root/f'assets/desk_muon_{g}_error.png'
        png(pixels,fit);png(bytes(min(255,abs(a-b)*16) for a,b in zip(pixels,target)),error)
        cards+=f'<div class="image-card"><h3>{names[g]}</h3><img alt="{names[g]}拟合" src="{embedded(fit)}" style="width:100%;margin:16px 0"><strong>{result["loss"]:.4f} MSE</strong><p>绝对像素误差 × 16</p><img alt="{names[g]}误差放大16倍" src="{embedded(error)}" style="width:100%"></div>'
    rows=''.join(f'<tr><td>{names[g]}</td><td>{r["steps"]:,}</td><td>{r["wall_median_ms"]/1000:.3f} s</td><td>{r["mse_median"]:.4f}</td></tr>' for g,r in groups.items())
    seeds=''.join('<tr><td>'+str(r['seed'])+'</td>'+''.join(f'<td>{r["mse"][g]:.4f}</td>' for g in names)+'</tr>' for r in holdout)
    tuning=''
    for source,data in [('desk_muon_pilot',pilot),('desk_muon_refine',refine)]:
        for r in data['runs']:
            if r['phase']!='tune':continue
            mode=['Adam','Muon N×6','Muon N×2 + Adam'][r['mode']]
            tuning+=f'<tr><td><a href="results/{source}/{r["file"]}">{mode}</a></td><td>{r["lr"]:g}</td><td>{r.get("schedule_floor",.05):.2f}</td><td>{r["steps"]:,}</td><td>{r["wall_ms"]/1000:.3f} s</td><td>{r["loss"]:.4f}</td></tr>'
    setting_rows=''.join(f'<li>{names[g]}：步长乘数 {r["lr"]:g}，{r["steps"]:,} 步，余弦末值 {r["floor"]:.2f}；GPU 中位数 {r["gpu_median_ms"]/1000:.3f} s，完整耗时范围 {r["wall_range_ms"][0]/1000:.3f}–{r["wall_range_ms"][1]/1000:.3f} s，MSE 范围 {r["mse_range"][0]:.4f}–{r["mse_range"][1]:.4f}。</li>' for g,r in groups.items())
    title='Muon 跑通了，<br>同耗时仍未胜过 Adam。' if adam_wins else 'Muon 的同耗时实测结果。'
    maxerr=max(r['relative_l2'] for r in audit['checks'])
    section=f'''<section class="section" id="desk-muon-results">
<div class="section-head"><div><div class="index">DESK / MUON · MEASURED</div><h2>{title}</h2></div><p>RTX 2080 SUPER · 320×128“周舒畅” · 1,600 高斯 × 8 链。两种矩阵分组、学习率网格与衰减曲线均实际测试；最终按完整进程耗时对齐。</p></div>
<div class="panel"><h3>同初值交错三轮 · 中位数</h3><div class="table-scroll"><table style="max-width:850px"><thead><tr><th>方案</th><th>迭代数</th><th>完整进程</th><th>全图 MSE ↓</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="caption">包含 CUDA 初始化、优化、读回、完整 FP64 重渲染、文件写入与退出；不含编译和 SSH。相对原参数 Adam 的完整耗时差：调参 Adam {time_delta['adam_tuned']:+.1f}%、N×6 Muon {time_delta['muon_joint']:+.1f}%、分组 Muon {time_delta['muon_grouped']:+.1f}%。这是近似等时预算，实际差异保留。</p></div>
<div class="gallery" style="margin-top:18px">{cards}</div><p class="caption">图片取各组 MSE 居中的一次，仅用于展示；表格分别报告耗时与 MSE 的中位数。评分沿用原版取整、截断、全图误差。</p>
<div class="panel" style="margin-top:18px"><h3>三组未参与调参的新初值</h3><div class="table-scroll"><table style="max-width:850px"><thead><tr><th>seed 基数</th><th>原 Adam</th><th>调参 Adam</th><th>Muon N×6</th><th>分组 Muon</th></tr></thead><tbody>{seeds}</tbody></table></div><p class="caption">每组仍为 8 条链：seed + 1009×链号。全部迭代数、步长、分组方式在运行这些种子前冻结；每组只跑一次。新初值下 N×6 Muon 实际耗时 4.10–4.22 秒，原 Adam 为 3.73–3.80 秒；Muon 用时更多仍未胜出。Adam 步长翻倍在这三个新种子上也未改善原配置，因此保留原默认步长。逐次耗时保留在 <a href="results/desk_muon_study/study.json">原始记录</a>。</p></div>
<details class="panel" style="margin-top:18px"><summary>Muon 如何对应高斯参数，以及正确性验证</summary>
<p>每条链的参数记为 p = [x, y, log σx, log σy, θ, a]，按 s = [0.15, 0.15, 0.025, 0.025, 0.025, 1] 组成 q = p/s；因此梯度乘 s，更新也乘 s。这与原 Adam 每类参数的基础步长一致。没有直接混合像素、弧度和亮度单位。</p>
<p>N×6 版本将整条链放进一个矩阵；分组版本把 [x,y]、[log σx,log σy] 放进两个 N×2，角度和亮度由 Adam 更新。两种方案都从同一个目标加权随机初始化开始，没有 MCMC。Muon 使用 momentum=0.95、Nesterov、5 次 Newton–Schulz、系数 (3.4445,−4.7750,2.0315)、原始 sqrt(max(1,N/K)) 缩放、零 weight decay；随后做相同的参数投影。</p>
<p>这是对高斯参数的实验性移植。<a href="https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py">Muon 原实现</a>面向神经网络隐藏权重矩阵，不能把其优势直接推广到此问题。Newton–Schulz 在这里近似正交化动量，不是在解拟合损失的 Hessian，也不等于 GN / LM。</p>
<p>本次全部使用 FP32。原实现的 BF16 正交化改为 FP32，不声称与 BF16 逐位一致或使用了 Tensor Core。N×6 只需 6×6 Gram，更新在单个 CUDA block 的共享内存完成；此处没有测半精度收益。</p>
<ul>{setting_rows}</ul>
<p>实际 GPU 更新与独立 CPU FP64、转置矩阵形式的 NS5 对照，覆盖 32 / 1,600 高斯、两种分组、第一步 / 多步动量，最大相对 L2 误差 {maxerr:.2e}。最终 {check['checked']} 份输出经原版 JavaScript 渲染器重算，像素完全一致，MSE 一致到 1e−9。<a href="results/desk_muon_pilot/update_check.json">更新校验</a> · <a href="results/desk_muon_study/renderer_check.json">全图校验</a>。</p>
</details>
<details class="panel" style="margin-top:18px"><summary>完整调参记录与复现</summary><p>在 seed=42 的 8 条链上调参；Adam 扫 0.5、1、2、4。Muon 第一轮扫 0.25–4，较小步长最好，于是补扫 0.03125、0.0625、0.125、0.25，并对比余弦末值 0.05 与 0。按各配置实际速度校准预算。选择后再校准一次时间、固定步数、交错重复三轮；随后运行三个新种子。</p>
<div class="table-scroll"><table style="max-width:850px"><thead><tr><th>方案 / 原始输出</th><th>步长</th><th>末值</th><th>步数</th><th>完整耗时</th><th>MSE</th></tr></thead><tbody>{tuning}</tbody></table></div>
<p>GPU 上其他工作负载保留，不能把当前测试的耗时与此前不同负载下的数据直接归因比较。这里同一可执行文件测全部优化器，主结论仅覆盖此目标、规模和测试过的配置。</p>
<pre>./run_muon_pilot.sh
./run_muon_refine.sh
./run_muon_study.sh
node src/check_muon.mjs
node src/check_muon_outputs.mjs
python3 src/finalize_report.py</pre>
<p><a href="src/muon_fit.cu">CUDA 源码</a> · <a href="source/desk_muon_study/muon_fit.cu">实测源码快照</a> · <a href="source/muon_provenance.json">上游版本</a> · <a href="results/desk_muon_study/summary.json">汇总 JSON</a>。连接 desk 需使用用户已有的 tmux 网络环境。</p></details></section>'''
    page=page.replace('<section class="section" id="desk-lm-results">',section+'<section class="section" id="desk-lm-results">',1)
    page=page.replace('<a href="#desk-lm-results">','<a href="#desk-muon-results">Muon</a><a href="#desk-lm-results">',1)
    return page
