"""Add the measured desk local-convexity experiment to the existing HTML report."""
import base64
import hashlib
import json
import statistics
import struct
import zlib


def update_lm_report(page, root):
    folder = root / 'results/desk_lm_study'
    if not (folder / 'study.json').exists():
        return page
    study = json.loads((folder / 'study.json').read_text())
    check = json.loads((folder / 'renderer_check.json').read_text())
    linear = json.loads((root / 'results/desk_lm/linearization_check.json').read_text())
    assert study['complete'] and check['passed'] and linear['passed']
    names = {'adam': '纯 Adam', 'block_lm': '分块 GN / LM', 'adam_then_lm': '短程 Adam → 分块 LM'}
    groups = {}
    for phase in ['repeat', 'holdout']:
        groups[phase] = {}
        for group in names:
            runs = [r for r in study['runs'] if r['phase'] == phase and r['group'] == group]
            assert len(runs) == 3
            groups[phase][group] = {
                'adam': runs[0]['adam_steps'], 'lm': runs[0]['lm_steps'],
                'wall_median_ms': statistics.median(r['wall_ms'] for r in runs),
                'wall_range_ms': [min(r['wall_ms'] for r in runs), max(r['wall_ms'] for r in runs)],
                'gpu_median_ms': statistics.median(r['gpu_ms'] for r in runs),
                'mse_median': statistics.median(r['loss'] for r in runs),
                'mse_range': [min(r['loss'] for r in runs), max(r['loss'] for r in runs)]}
    # Select the median-MSE representative; this selection is explicitly visual only.
    representatives = {g: sorted([r for r in study['runs'] if r['phase'] == 'repeat' and r['group'] == g], key=lambda r: r['loss'])[1]['file'] for g in names}
    paired = []
    for seed in [1042, 2042, 3042]:
        runs = {r['group']: r for r in study['runs'] if r['phase'] == 'holdout' and r['seed'] == seed}
        paired.append({'seed_base': seed, 'mse': {g: runs[g]['loss'] for g in names}, 'wall_ms': {g: runs[g]['wall_ms'] for g in names}, 'adam_best': all(runs['adam']['loss'] < runs[g]['loss'] for g in ['block_lm', 'adam_then_lm'])})
    assert all(r['adam_best'] for r in paired)
    for source, digest in study['sources'].items():
        assert hashlib.sha256((root/'source/desk_lm_study'/source.split('/')[-1]).read_bytes()).hexdigest() == digest
    summary = {'groups': groups, 'paired_holdout': paired, 'representatives': representatives,
               'study': study, 'linearization_check': linear, 'renderer_checks': check,
               'calibration_note': 'Frozen budgets 3456 and 3120 match rounding the time-only estimates to multiples of 16. The original study.json rule string says 32, a documentation typo; raw measurements and actual settings are preserved. The current reproduction script corrects that description.',
               'conclusion': 'The tested block-diagonal damped Gauss-Newton implementation loses to pure Adam at similar total process time on this target. This does not rule out coupled or matrix-free global Gauss-Newton, nor an amplitude-only convex subproblem.'}
    (folder/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')

    def png(pixels, path):
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind+data) & 0xffffffff)
        raw = b''.join(b'\0'+pixels[y*320:(y+1)*320] for y in range(128))
        path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>2I5B',320,128,8,0,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))

    def embedded(path):
        return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()

    target = bytes(json.loads((root/'assets/zhou_target.json').read_text())['pixels'])
    cards = ''
    for group, name in names.items():
        result = json.loads((folder/representatives[group]).read_text())
        raw = (folder/representatives[group].replace('.json','.pgm')).read_bytes()
        header = b'P5\n320 128\n255\n'
        assert raw.startswith(header)
        pixels = raw[len(header):]
        fit = root/f'assets/desk_lm_{group}.png'
        error = root/f'assets/desk_lm_{group}_error.png'
        png(pixels,fit)
        png(bytes(min(255,abs(a-b)*16) for a,b in zip(pixels,target)),error)
        cards += f'<div class="image-card"><h3>{name}</h3><img alt="{name}拟合" src="{embedded(fit)}" style="width:100%;margin:16px 0"><strong>{result["loss"]:.4f} MSE</strong><p>绝对像素误差 × 16</p><img alt="{name}误差放大16倍" src="{embedded(error)}" style="width:100%;margin-top:12px"></div>'
    primary = groups['repeat']
    rows = ''.join(f'<tr><td>{names[g]}</td><td>{r["adam"]:,} + {r["lm"]:,}</td><td>{r["wall_median_ms"]/1000:.3f} s</td><td>{r["mse_median"]:.4f}</td></tr>' for g,r in primary.items())
    seed_rows = ''.join(f'<tr><td>{r["seed_base"]}</td><td>{r["mse"]["adam"]:.4f}</td><td>{r["mse"]["block_lm"]:.4f}</td><td>{r["mse"]["adam_then_lm"]:.4f}</td></tr>' for r in paired)
    baseline = primary['adam']['wall_median_ms']
    gaps = [100*(primary[g]['wall_median_ms']/baseline-1) for g in ['block_lm','adam_then_lm']]
    pilots = []
    for path in sorted((root/'results/desk_lm').glob('*.json')):
        if path.stem.startswith(('pilot_','sweep_')):
            r = json.loads(path.read_text())
            seconds = float(path.with_suffix('.time').read_text().strip().split('=')[1])
            pilots.append((r['adam_steps'],r['lm_steps'],seconds,r['loss']))
    pilot_rows = ''.join(f'<tr><td>{a:,} + {l:,}</td><td>{t:.2f} s</td><td>{mse:.4f}</td></tr>' for a,l,t,mse in pilots)
    section = f'''<section class="section" id="desk-lm-results">
<div class="section-head"><div><div class="index">DESK / LOCAL CONVEX APPROXIMATION · MEASURED</div><h2>局部凸近似跑通了，<br>同耗时仍然是 Adam 更好。</h2></div><p>RTX 2080 SUPER · 320×128“周舒畅” · 1,600高斯 × 8条链 · FP32。每颗高斯解6×6阻尼方程，再按整条链的真实连续损失接受或拒绝。没有使用MCMC。</p></div>
<div class="panel"><h3>约3.7–3.9秒完整进程预算 · 同初值交错3轮</h3><div class="table-scroll"><table style="max-width:850px"><thead><tr><th>方案</th><th>Adam + LM 步数</th><th>总耗时中位数</th><th>原版MSE中位数</th></tr></thead><tbody>{rows}</tbody></table></div>
<p><strong class="green">结论：保留纯Adam。</strong>分块LM与短程Adam接LM的中位耗时相对基线为 {gaps[0]:+.2f}% / {gaps[1]:+.2f}%，但MSE均更高。固定初值的CUDA原子加法仍会引起微小轨迹差异，因此报告3次质量与时间分布。</p>
<p class="caption">计时在desk的Python父进程中包住完整子进程：含CUDA上下文、初始化、计算、回传、FP64全图验证、写文件、清理与退出；不含编译和SSH。全部从随机初始化开始，GPU图形负载保持原状。步数先按独立试跑的耗时校准，然后冻结；没有按正式结果重新挑选预算。</p></div>
<div class="gallery" style="grid-template-columns:repeat(3,1fr);margin-top:18px">{cards}</div>
<p class="caption">图片选择各方案三次重复中的MSE中位样本，作为视觉对照；时间和MSE中位数分别统计，不一定来自同一次运行。下排误差统一放大16倍。</p>
<div class="two" style="margin-top:18px"><div class="panel"><h3>另外3组初始种子，结论仍一致</h3><div class="table-scroll"><table><thead><tr><th>种子基数</th><th>Adam</th><th>LM</th><th>Adam → LM</th></tr></thead><tbody>{seed_rows}</tbody></table></div><p class="caption">每组含8条链，种子为base + 1009×链号。1042 / 2042 / 3042未参与预算选择；每组每方案运行一次并轮换顺序。它们是初始化泛化检查，不是额外的三次固定种子计时。</p></div>
<div class="panel"><h3>这一版“凸”在哪里</h3><p>局部模型为 Hⱼ = JⱼᵀJⱼ，解 (Hⱼ + λDⱼ)Δⱼ = −Jⱼᵀr；Dⱼ为有正下界的对角缩放。每个阻尼二次子问题是凸的，联合高斯拟合仍非凸。</p><p>实现保留6σ ROI和解析Jacobian，使用FP32 Cholesky；求解后限步并投影到参数边界。每条链同时试更新全部高斯，重新渲染；只有6σ近似渲染下的连续全图MSE下降才接受，否则增大阻尼。</p><p>这是一种分块、LM式阻尼方法，省略了不同高斯之间的JᵀJ交叉项；不等于精确求解全局LM子问题，也没有测试独立的幅度凸优化。</p></div></div>
<div class="two" style="margin-top:18px"><div class="panel"><h3>验证结果</h3><ul><li>32个实际GPU局部系统，与独立Float64全图中心差分比较：梯度最大相对L2误差 {linear['max_gradient_relative_l2']:.2e}，Hessian {linear['max_hessian_relative_l2']:.2e}，更新步最大绝对差 {linear['max_step_abs']:.2e}。</li><li>全部19次运行（含1次预热）的参数交回原版renderer / loss：MSE一致、0个不同像素。</li><li>程序检查每一步、每条链的LM连续损失均不增加；外部脚本再检查导出的曲线和接受/拒绝计数。</li><li>优化连续未取整、未截断的强度误差，最终验收仍是原版round / clip MSE；两者不能混同。</li></ul></div>
<div class="panel"><h3>为什么没有赢：目前的判断</h3><p>单步需要累计6维梯度和21个Hessian元素，并额外渲染候选、验收和提交。局部曲率没有换来足够多的迭代节省。</p><p>重叠高斯之间的耦合被分块近似省略，是后续值得验证的原因。它目前是解释假设，不是已完成的归因消融。矩阵自由的全局Jv / Jᵀv求解可能改善这一点，但本次没有性能结论。</p><p>先前设想的“短Adam后接LM精修”在当前预算也没有优势；继续采用纯Adam更有实测依据。</p></div></div>
<details style="margin-top:18px"><summary>所有探索性试跑，包括较长Adam预热</summary><div class="table-scroll"><table style="max-width:850px"><thead><tr><th>Adam + LM</th><th>完整进程</th><th>MSE</th></tr></thead><tbody>{pilot_rows}</tbody></table></div><p class="caption">探索阶段各一次，/usr/bin/time仅保留两位小数；与上方高分辨率父进程计时分开。即使6,000 Adam + 900 LM，约3.78秒 / MSE 1.4883，也未胜过纯Adam基线。</p></details>
<div class="panel" style="margin-top:18px"><h3>复现</h3><pre class="code">./run_lm_study.sh
# 已编译后，在desk项目目录单独运行：
./lm_fit --gaussians 1600 --chains 8 --adam-steps 0 --lm-steps 3456 --output results/lm_replay
# 本地独立复核：
node src/check_lm.mjs
node src/check_lm_outputs.mjs</pre><p class="caption">若当前终端无法直连desk，可沿用本项目README里的tmux run-shell入口。归档了实测源码和SHA-256；没有调整驱动、时钟或其他GPU进程。</p></div>
<p class="caption"><a href="results/desk_lm_study/summary.json">完整汇总 / 原始记录 / 验证</a> · <a href="results/desk_lm_study/study.json">19次运行清单</a> · <a href="src/lm_fit.cu">CUDA实现</a> · <a href="run_lm_study.sh">复现脚本</a> · <a href="https://ceres-solver.readthedocs.io/latest/nnls_solving.html">非线性最小二乘方法参考</a></p></section>'''
    page = page.replace('<section class="section" id="ablation-results">', section + '<section class="section" id="ablation-results">', 1)
    page = page.replace('<a href="#ablation-results">消融</a>', '<a href="#desk-lm-results">desk凸近似</a><a href="#ablation-results">消融</a>', 1)
    return page
