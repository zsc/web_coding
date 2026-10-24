"""Publish the measured optimizer ablation, including every timed repetition."""
import base64
import hashlib
import json
import statistics
import struct
import zlib


def update_ablation_report(page, root):
    path = root / 'results/webgpu_ablation.json'
    if not path.exists():
        return page
    study = json.loads(path.read_text())
    check = json.loads((root / 'results/webgpu_ablation_check.json').read_text())
    assert study['complete'] and check['passed'] and len(study['runs']) == 9
    names = {'hybrid': 'MCMC → Adam', 'adam_equal_steps': '纯 Adam · 同步数',
             'adam_equal_time': '纯 Adam · 同总耗时'}
    groups = {}
    for group in names:
        runs = [r for r in study['runs'] if r['group'] == group]
        assert len(runs) == 3 and all(not r['cancelled'] and not r['cold_pipeline'] for r in runs)
        groups[group] = {
            'mcmc': runs[0]['mcmc'], 'adam': runs[0]['adam'],
            'total_median_ms': statistics.median(r['total_ms'] for r in runs),
            'total_min_ms': min(r['total_ms'] for r in runs),
            'total_max_ms': max(r['total_ms'] for r in runs),
            'gpu_median_ms': statistics.median(r['gpu_ms'] for r in runs),
            'mse_median': statistics.median(r['loss'] for r in runs),
            'mse_min': min(r['loss'] for r in runs), 'mse_max': max(r['loss'] for r in runs)}
    h, p, matched = [groups[g] for g in names]
    equal_time_delta = 100 * (matched['total_median_ms'] / h['total_median_ms'] - 1)
    mse_reduction = 100 * (1 - matched['mse_median'] / h['mse_median'])
    same_steps_faster = 100 * (1 - p['total_median_ms'] / h['total_median_ms'])
    summary = {
        'settings': study['settings'], 'groups': groups,
        'equal_time_wall_difference_pct': equal_time_delta,
        'equal_time_mse_reduction_pct': mse_reduction,
        'equal_steps_wall_reduction_pct': same_steps_faster,
        'calibration': study['calibration'], 'calibrated_adam_steps': study['calibrated_adam_steps'],
        'runs': study['runs'], 'validation': check,
        'source_sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in [
            'source/webgpu_ablation/hybrid_webgpu.mjs', 'source/webgpu_ablation/hybrid_playground.mjs',
            'source/webgpu_ablation/hybrid_playground.html']},
        'target_sha256': hashlib.sha256((root / 'assets/zhou_target.pgm').read_bytes()).hexdigest(),
        'limits': 'One fixed target and the same eight seeded initial chains. Repetitions measure timing variability, not independent seed ensembles. Warm pipelines; full Float64 verification included. Cosine schedule normalized to each total Adam length.'}
    (root / 'results/webgpu_ablation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    def png(pixels, path):
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
        raw = b''.join(b'\0' + bytes(pixels[y*320:(y+1)*320]) for y in range(128))
        path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', 320, 128, 8, 0, 0, 0, 0))
                         + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))

    cards = ''
    for group, name in names.items():
        r = study['representatives'][group]
        image_path = root / f'assets/ablation_{group}.png'
        png(r['image'], image_path)
        uri = 'data:image/png;base64,' + base64.b64encode(image_path.read_bytes()).decode()
        cards += f'<div class="image-card"><h3>{name}</h3><img alt="{name}拟合周舒畅" src="{uri}" style="width:100%;margin:16px 0"><strong>{r["loss"]:.4f} MSE</strong><p>{r["mcmc"]:,} MCMC + {r["adam"]:,} Adam</p></div>'
    rows = ''.join(f'<tr><td>{names[g]}</td><td>{r["mcmc"]:,} + {r["adam"]:,}</td><td>{r["gpu_median_ms"]/1000:.3f} s</td><td>{r["total_median_ms"]/1000:.3f} s</td><td>{r["mse_median"]:.4f}</td></tr>' for g, r in groups.items())
    raw_rows = ''.join(f'<tr><td>{r["round"]+1}</td><td>{names[r["group"]]}</td><td>{r["total_ms"]/1000:.4f} s</td><td>{r["loss"]:.7f}</td></tr>' for r in study['runs'])
    section = f'''<section class="section" id="ablation-results">
<div class="section-head"><div><div class="index">NEW / OPTIMIZER ABLATION</div><h2>这个案例，MCMC 可以省掉。</h2></div><p>同一个“周舒畅”、1,600高斯、8条链和FP32。移除MCMC后，8,000步Adam质量几乎不变；将省下的时间给Adam，MSE再降 {mse_reduction:.2f}%。</p></div>
<div class="panel"><h3>同场交错3轮 · 中位数</h3><div class="table-scroll"><table><thead><tr><th>方案</th><th>MCMC + Adam 步数</th><th>GPU时间</th><th>总耗时</th><th>完整MSE</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="caption">总耗时包含初始化、GPU计算、回传、Worker启动和Float64全图验证；均复用已预热管线。三组轮换顺序，每轮从相同的8条初始链重新开始。校准、预热与两次前期试跑不混入这9次统计。</p>
<p class="caption">三次耗时范围依次为 {h['total_min_ms']/1000:.3f}–{h['total_max_ms']/1000:.3f} / {p['total_min_ms']/1000:.3f}–{p['total_max_ms']/1000:.3f} / {matched['total_min_ms']/1000:.3f}–{matched['total_max_ms']/1000:.3f} 秒。每组3次MSE均相同；小幅计时差不视为跨场景的稳定加速保证。</p>
<p><strong class="green">同步数：</strong>纯Adam快 {same_steps_faster:.2f}%，MSE仅增加 {p['mse_median']-h['mse_median']:.7f}（{(p['mse_median']/h['mse_median']-1)*100:.3f}%）。<br><strong class="green">同总耗时：</strong>纯Adam校准为 {matched['adam']:,} 步，总耗时差 {equal_time_delta:+.2f}%，MSE从 {h['mse_median']:.4f} 降至 {matched['mse_median']:.4f}。没有插入等待来凑时间。</p></div>
<div class="gallery" style="grid-template-columns:repeat(3,1fr);margin-top:18px">{cards}</div>
<div class="two" style="margin-top:18px"><div class="panel"><h3>为什么会这样</h3><p>当前初始化已经按目标笔画亮度分配高斯，给Adam提供了不错的起点。10,000次MCMC每次只尝试一个高斯的一个参数；Adam每步并行更新全部高斯。对当前8,000步细化预算，MCMC提前搜索带来的最终质量差异已经很小。</p><p>所以先前“先MCMC找位置，再Adam细化”的解释只是设计动机；这次对照表明，在此配置中它不是达到低MSE的必要条件。此目标优先用纯Adam更合理。</p></div>
<div class="panel"><h3>如何公平比较，以及结论边界</h3><ul><li>先完整预热，再测混合基线与纯Adam的8,000 / 10,000步耗时，用线性插值确定 {matched['adam']:,} 步；校准没有使用MSE。</li><li>三组采用相同Adam超参数，余弦学习率按各自总步数归一化；延长组从头训练，并非在8,000步结果上续跑。</li><li>8条链种子固定。3次重复是计时重复，不是3组独立初始化；结论限于此目标、初始化和M5浏览器。</li><li>三组代表参数交给原版完整renderer / loss复核，MSE完全一致，每组0个不同像素。desk未在本次重跑消融。</li></ul></div></div>
<details style="margin-top:18px"><summary>展开全部9次结果（实际执行顺序）</summary><div class="table-scroll"><table><thead><tr><th>轮次</th><th>方案</th><th>总耗时</th><th>MSE</th></tr></thead><tbody>{raw_rows}</tbody></table></div></details>
<p class="caption">重现：下方试跑区新增“消融：纯 Adam / 同步数 / 同耗时”按钮；保留当前10,000 + 8,000基线设置即可。单独重现纯Adam同耗时配置，将MCMC设为0、Adam设为{matched['adam']:,}。其他设备的同耗时步数应重新校准。<a href="#playground">去试跑 →</a></p>
<p class="caption"><a href="results/webgpu_ablation_summary.json">汇总与源码SHA-256</a> · <a href="results/webgpu_ablation.json">完整校准、9次运行与三组参数</a> · <a href="results/webgpu_ablation_check.json">原版渲染复核</a> · <a href="src/check_ablation.mjs">独立复核脚本</a></p></section>'''
    page = page.replace('<section class="section" id="final-result">', section + '<section class="section" id="final-result">', 1)
    page = page.replace('<a href="#webgpu-results">实测</a>', '<a href="#ablation-results">消融</a><a href="#webgpu-results">实测</a>', 1)
    page = page.replace('下方试跑区可直接重现。', '新补消融：同总耗时纯Adam达到MSE 1.310；下方保留基线与对照入口。', 1)
    return page
