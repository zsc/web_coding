"""Recorded results of the explicitly requested 1node x16PPU jobs."""
import html
import json
from pathlib import Path


def section(root):
    root=Path(root)
    base=root/'results/rgb_16ppu'
    if not (base/'jobs.json').exists():return ''
    jobs=json.loads((base/'jobs.json').read_text())
    completion=''
    if (base/'completion_summary.json').exists():
        finished=json.loads((base/'completion_summary.json').read_text())['runs']
        finalrows=[]
        for run in finished:
            q=next(q for q in run['rows'] if q['training_image_draws']==5120000 and q['weights_prefix']=='slow_ema')
            name=('U-Net' if run['run'].startswith('unet') else '大 JiT')+' / '+('5e−5' if '5e5' in run['run'] else '2e−4')
            finalrows.append(f'<tr><td>{name}</td><td>{q["kid_mean"]*1000:.2f} ± {q["subset_std"]*1000:.2f}</td><td>{run["median_images_per_second"]:,.0f}</td><td>{run["pipeline_seconds"]/60:.2f} 分</td><td>{run["skip"]} / {run["updates"]}</td></tr>')
        completion='''<div class="card"><h3>本轮已完成 · 最终对照</h3><p>四组任务均成功，累计各 512 万张次，最终模型、双 EMA 与 AdamW 的 16 份副本逐位一致。下表统一比较最终慢 EMA（每 32 张次衰减 0.9999），不跨检查点挑最低值。</p><div class="scroll"><table><thead><tr><th>模型 / 学习率</th><th>最终慢 EMA KID×1,000 ↓</th><th>稳态中位吞吐 · 张/秒</th><th>本轮成功任务墙钟</th><th>AMP 跳过 / 新增步数</th></tr></thead><tbody>'''+''.join(finalrows)+'''</tbody></table></div><p class="caption">墙钟包含本次成功尝试的数值检查、初始化、续训、存盘与阶段评估；不包含先前断点训练、排队和失败重试。U-Net 本轮新增 320 万张次，JiT 新增 256 万张次；不是从零训练到 512 万的总耗时。吞吐取各 100 步训练块的中位数，去掉首块，排除评估与存盘。</p><p>当前配方下 U-Net 的 KID 较低；学习率之间的小差距不能视为显著优势。后半程验证 loss 持续下降，但 KID 变化小，显示继续采用相同配方加训练量的收益已减弱。检查过两种架构的最终慢 EMA 前 64 张未筛选样本，仍有局部五官变形与模糊。</p><img class="gallery" src="results/rgb_16ppu/final_curves.svg" alt="16卡四组最终慢EMA验证损失与KID曲线"><p class="caption">相同慢 EMA 口径；误差线是 KID 子集标准差，不是置信区间。只连接实际评估点，没有平滑或外推。</p><p class="small"><a href="results/rgb_16ppu/completion_summary.json">完成记录与逐点指标</a></p></div>'''
    rows=[];quality=[];previews=[];evidence=[]
    for run in jobs['runs']:
        d=root/run['out']
        def read(name):
            p=d/name
            return json.loads(p.read_text()) if p.exists() else {}
        pipeline=read('pipeline.json');training=read('training.json');gate=read('numerical_gate.json')
        status={'checking':'16 卡数值验证中','training':'训练中','evaluating':'评估中','complete':'完成','failed':'失败','failed_timeout':'超时'}.get(training.get('status',pipeline.get('status')),'等待调度')
        if pipeline.get('status','').startswith('failed'):status='失败'
        name=('U-Net' if run['kind']=='unet' else '大 JiT')+f' · lr={run["lr"]:g}'
        images=training.get('training_image_draws',1920000 if run['kind']=='unet' else 2560000)
        speed=f'{training["images_per_second"]:,.0f}' if 'images_per_second' in training else '—'
        rows.append(f'<tr><td>{name}</td><td>{run["task_id"]}</td><td>1 × 16</td><td>{images/1e6:.3f} / 5.120 M</td><td>{speed}</td><td>{"通过" if gate.get("passed") else "待完成"}</td><td>{status}</td></tr>')
        if gate:
            evidence.append(f'<a href="{run["out"]}/numerical_gate.json">{name} 数值记录</a>')
        results=read('evaluation/evaluation.json').get('rows',[])
        for q in results:
            quality.append(f'<tr><td>{name}</td><td>{q["training_image_draws"]/1e6:.3f} M</td><td>{q["weights_prefix"]}</td><td>{q["kid_mean"]*1000:.2f} ± {q["subset_std"]*1000:.2f}</td></tr>')
        if results:
            q=next((q for q in reversed(results) if q['weights_prefix']=='slow_ema'),results[-1])
            img=d/'evaluation'/f'{q["name"]}_64.png'
            regular=next((r for r in results if r['weights_prefix']=='ema' and r['training_image_draws']==q['training_image_draws']),q)
            regular_img=d/'evaluation'/f'{regular["name"]}_64.png'
            if img.exists():previews.append(f'<div><h3>{name} · {q["training_image_draws"]/1e6:.3f} M</h3><img class="gallery" data-final-preview data-slow="{img.relative_to(root)}" data-regular="{regular_img.relative_to(root)}" src="{img.relative_to(root)}" alt="慢 EMA · 最终固定前64张未筛选 RGB 样本"></div>')
    return '''<section id="training-16ppu"><div class="num">FINAL RESULTS / 1 NODE × 16 PPU PER EXPERIMENT</div>
<h2>四组 16 卡训练完成，各累计 512 万张次</h2>
<p>四组 RGB64 实验：U-Net 两条原分支分别从 60k 续训；大 JiT 两条分支都从稳定的 batch128 / lr2e−4 / 20k 断点分出，比较 lr5e−5 与 lr2e−4。每组分配单机 16 张 PPU，global batch 512、local batch 32，四组均已成功完成并释放任务资源。每组均在 16 卡数值检查通过后开始训练。</p>
'''+completion+'''<div class="card"><div class="scroll"><table><thead><tr><th>实验</th><th>任务</th><th>节点 × PPU</th><th>累计看图量 / 目标</th><th>最后训练块 · 张/秒</th><th>数值门槛</th><th>状态</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div></div>
<p>本轮是新的大 batch 分支：继承模型、AdamW、scaler 和 EMA；因卡数和 batch 改变，明确使用新的每卡随机流。EMA 衰减按看图量调整为 0.999¹⁶ 和 0.9999¹⁶。U-Net 保留原慢 EMA；JiT 的慢 EMA 从本轮起点初始化。原 batch32 的 80k 任务完成后保留作对照。</p>
<p>训练 Graph 捕获每卡前向和反向，梯度在图外以 FP32 求平均，再做全局裁剪与 AdamW；任何卡溢出会传递到所有卡。JiT 另启用已验证的 25 个 Triton RMSNorm。数值门槛比较完整 batch 的 FP32 梯度、12 次优化器更新、双 EMA、scaler 增长和单卡 Inf；检查 16 份模型及优化器状态完全一致。</p>
<div class="bar" style="margin-top:24px"><label for="finalPreviewEma">最终未筛选样本　<select id="finalPreviewEma"><option value="slow_ema">慢 EMA · 与最终对照表一致</option><option value="ema">常规 EMA · 相同初始噪声</option></select></label></div>
<p class="caption">默认展示最终慢 EMA，每组固定种子的前 64 张；全部 512 张均参与 KID。可切换到相同终点的常规 EMA。</p>
<div class="cols">'''+''.join(previews)+'''</div><h3>所有阶段与两份 EMA 的质量记录</h3><div class="scroll"><table><thead><tr><th>实验</th><th>累计看图量</th><th>EMA</th><th>KID×1,000 ↓</th></tr></thead><tbody>'''+(''.join(quality) or '<tr><td colspan="4">等待阶段评估；所有结果保留，未获得质量结果前不宣称大 batch 更优。</td></tr>')+'''</tbody></table></div>
<p class="small">仍为固定 512 张生成图 / 2,048 张真实参考 / 50 个 KID 子集。32 个采样种子分给 16 卡，按原顺序汇总，并检查首尾种子的并行/串行结果。评估使用原始模型实现。± 为子集标准差，不是置信区间。U-Net 与 JiT 起点及新增看图量不同；两种架构的终点只对齐累计看图量。训练吞吐是新配方实测，不能与 batch32 的步耗时直接当作纯加速比。</p>
<details><summary>16 卡启动验证与重试记录</summary><p class="small">低学习率 U-Net 的初始跨卡 FP32 参考检查两次超限，分别为 4.51e−5 和 6.06e−4 相对误差；FP64 归约检查通过。将参考计算固定为确定性卷积并关闭 TF32 后，误差为 9.26e−8，重新通过最初的 2e−5 门槛。此设置只用于参考检查；正式 AMP 训练恢复原配置，12 次更新的 Graph / eager 参数、双 EMA 和 AdamW 状态逐位一致。两项参考设置一起改变，未单独归因。</p><p class="small">低学习率 JiT 首次在数值检查启动阶段发生原生 SIGSEGV；保留日志，重试后数值检查和训练通过，未确认该底层崩溃根因。所有失败任务及冻结源码都保留在任务清单的 previous_attempts 中。</p><div class="linkrow small">'''+''.join(evidence)+'''</div></details>
<div class="linkrow small"><a href="results/rgb_16ppu/jobs.json">任务与命令</a><a href="src/run_16ppu.py">启动入口</a><a href="src/distributed_acceleration.py">多卡同步实现</a><a href="profiling.html">单卡加速依据</a></div></section>'''
