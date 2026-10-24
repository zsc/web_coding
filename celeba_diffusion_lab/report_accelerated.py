"""Completed single-PPU U-Net continuations, retained as historical controls."""
import html, json
from pathlib import Path


def section(root):
    root=Path(root);base=root/'results/rgb_accelerated80k'
    if not (base/'jobs.json').exists():return ''
    jobs=json.loads((base/'jobs.json').read_text());rows=[];quality=[];previews=[]
    for run in jobs['runs']:
        d=root/run['out'];p=d/'pipeline.json';pipeline=json.loads(p.read_text()) if p.exists() else {}
        p=d/'training.json';training=json.loads(p.read_text()) if p.exists() else {}
        step=training.get('step',60000);last={}
        for p in sorted(d.glob('training_*.log')):
            for line in p.read_text(errors='replace').splitlines():
                try:r=json.loads(line)
                except ValueError:continue
                if r.get('step',0)>=step and 'loss' in r:step=r['step'];last=r
        status={'training':'训练中','evaluating':'评估中','complete':'完成','failed':'失败'}.get(pipeline.get('status'),'已提交')
        label='低学习率' if run['lr']==5e-5 else '原学习率对照'
        loss=f"{last['loss']:.5f}" if last else '—'
        rows.append(f'<tr><td>{label} · lr={run["lr"]:g}</td><td>{pipeline.get("task_id","—")}</td><td>{step:,} / 80,000</td><td>{loss}</td><td>{status}</td></tr>')
        p=d/'evaluation/evaluation.json'
        if p.exists():
            for q in json.loads(p.read_text())['rows']:
                quality.append(f'<tr><td>{label}</td><td>{q["step"]:,}</td><td>{"0.9999" if q["weights_prefix"]=="slow_ema" else "0.999"}</td><td>{q["kid_mean"]*1000:.2f} ± {q["subset_std"]*1000:.2f}</td></tr>')
        images=sorted(d.glob('ema_step*.png'),key=lambda p:int(p.stem.replace('ema_step','')))
        if images:
            p=images[-1];previews.append(f'<div><h3>{label} · {p.stem.replace("ema_step","")} 步</h3><img class="gallery" src="{p.relative_to(root)}" alt="加速续训固定噪声预览"></div>')
    return '''<section id="accelerated-training"><div class="num">HISTORICAL CONTROL / RGB U-NET / TRAINING CUDA GRAPH</div>
<h2>单卡对照已完成：60k → 80k</h2><p>两条分支各使用 1 张 PPU，batch 32；保留各自 60k 的完整模型、优化器、scaler、双 EMA 与随机状态。主分支 lr=5e−5，对照 lr=2e−4。训练 Graph 已通过真实断点 100 步逐位一致检查。</p>
<p>60k 的常规 EMA KID×1,000：低学习率 15.01 ± 2.00，对照 15.73 ± 2.10；慢 EMA 分别为 15.43 / 18.04。差值来自同一训练种子的固定评估，不能单凭它断言学习率优劣。70k、80k 的双 EMA 评估均已完成。80k 常规 EMA 为 14.57 / 15.53，慢 EMA 为 14.20 / 16.09；这些单卡 batch32 分支保留作对照。</p>
<div class="card"><div class="scroll"><table><thead><tr><th>分支</th><th>Sofy 任务</th><th>最新日志步数 / 目标</th><th>累计训练桶 loss</th><th>阶段</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div><p class="caption">每条新增 64 万张次，80k 总计 256 万张次。阶段评估在同一张已分配 PPU 上执行，不额外占卡等待。训练与阶段评估均已完成；以下保留最终记录。</p></div>
<div class="cols">'''+''.join(previews)+'''</div><div class="scroll"><table><thead><tr><th>分支</th><th>步数</th><th>EMA</th><th>KID×1,000 ↓</th></tr></thead><tbody>'''+(''.join(quality) or '<tr><td colspan="4">等待 70k 检查点；采样及 KID 已编入训练任务。</td></tr>')+'''</tbody></table></div>
<p class="small">RGB64 / 固定 512 张生成图 / 2,048 张真实参考 / Heun 50 区间；与上一轮同一协议。± 为子集标准差。</p><div class="linkrow small"><a href="results/rgb_accelerated80k/jobs.json">本轮任务记录</a><a href="profiling.html">加速依据与数值验证</a><a href="src/run_accelerated_continuation.py">续训与阶段评估入口</a></div></section>'''
