"""Current conclusions, kept separate from the historical experiment sections."""
import json
from pathlib import Path


def section(root):
    root = Path(root)
    complete = root / 'results/rgb_16ppu/completion_summary.json'
    if not complete.exists():
        return ''
    runs = json.loads(complete.read_text())['runs']
    final = {
        r['run']: next(q for q in r['rows']
                       if q['training_image_draws'] == 5120000 and q['weights_prefix'] == 'slow_ema')
        for r in runs
    }
    hardware = json.loads((root / 'results/device_comparison_rgb.json').read_text())
    acceleration = json.loads((root / 'results/ppu_acceleration/summary.json').read_text())['models']
    u = acceleration['unet']['modes']['graph']['paired_speedup_median']
    j = acceleration['large_jit']['modes']['graph_triton']['paired_speedup_median']
    uk = [q['kid_mean'] * 1000 for k, q in final.items() if k.startswith('unet')]
    jk = [q['kid_mean'] * 1000 for k, q in final.items() if k.startswith('jit')]
    return f'''<section id="summary"><div class="num">SUMMARY / COMPLETED RGB EXPERIMENTS / 2026-10-24</div>
<h2>目前有效的改进，以及证据的边界</h2>
<div class="cols">
<div class="card"><h3>先把像素空间 U-Net 训练充分</h3>
<p>U-Net 和 JiT 都直接预测 RGB 干净图像，使用相同 flow 目标与 Heun 采样，没有 VAE。U-Net 在原 batch 32、lr 2e−4 配方下，从 5k 续到 40k，KID×1,000 从 <strong>46.25 降到 19.08</strong>。这支持早期训练不足；切换到 Transformer 并非必要条件。</p>
<p>本轮累计 512 万张次的最终慢 EMA：U-Net 为 <strong>{min(uk):.2f}–{max(uk):.2f}</strong>，32.63M JiT 为 <strong>{min(jk):.2f}–{max(jk):.2f}</strong>。当前配方下 U-Net 更好，但两种架构的训练历史、batch 日程与慢 EMA 起点不同，不能当作只改变架构的严格消融。</p>
<p class="small"><a href="#unet-continuation">同配方 5k→40k 的对照</a> · <a href="#training-16ppu">四组最终结果与曲线</a></p></div>
<div class="card"><h3>EMA 和引导有收益，学习率需实测</h3>
<p>最终四组慢 EMA 的 KID 均低于各自常规 EMA，差距约 0.38–0.78；这是同一训练种子的数值趋势。低学习率并非普遍更优：最终 U-Net 的 lr 2e−4 数值较好，JiT 的 lr 5e−5 较好，两档差距都不足以支持显著性结论。</p>
<p>较早的 U-Net20k 配合 5k 弱模型，在 t∈[0.1,0.8]、强度 1.6 下，KID 从 <strong>23.48 降到 18.76</strong>。强度 2.0 虽降到 17.91，局部色块与对比更差。当前实现前向次数从 99 增至 198；尚未在最终 512 万张次权重上验证。</p>
<p class="small"><a href="#round2">学习率、双 EMA 与引导完整记录</a></p></div>
</div>
<div class="card" style="margin-top:24px"><h3>加速收益按三种口径分别看</h3>
<div class="scroll"><table><thead><tr><th>比较</th><th>实测结果</th><th>适用范围</th></tr></thead><tbody>
<tr><td>1 PPU / RTX 2080 SUPER</td><td>训练 {hardware['training_speedup']:.2f}×；Graph 采样 {hardware['graph_sampling_speedup']:.2f}×</td><td>同代码、权重、输入，小 RGB JiT / batch 32；实际设备与厂商运行时组合。</td></tr>
<tr><td>单 PPU 训练实现优化</td><td>U-Net Graph {u:.2f}×；大 JiT Graph + Triton {j:.2f}×</td><td>batch 32，五轮独立进程配对复测的中位加速；含数据与一份 EMA，排除初始化、验证、存盘。</td></tr>
<tr><td>每组 1 节点 × 16 PPU</td><td>U-Net 约 1.25 万张/秒；JiT 约 1.07 万张/秒</td><td>global batch 512、双 EMA；是新配方吞吐，不能当作固定 batch 的纯加速比。</td></tr>
</tbody></table></div>
<p>本轮成功任务墙钟为 <strong>5.85–6.66 分钟</strong>（约 94–107 PPU·分钟），包括数值检查、续训与评估。U-Net 从 192 万张次追加 320 万，JiT 从 256 万追加 256 万；此前训练、排队与失败重试均不在这段计时内。三行倍率不能相乘。</p>
<p class="small"><a href="profiling.html">ASight 时间线、Triton 与数值验证</a> · <a href="results/rgb_16ppu/completion_summary.json">最终指标与实际耗时</a></p></div>
<div class="cols" style="margin-top:24px">
<div class="card"><h3>哪些判断还不成立</h3><p>固定总 batch 32 的 U-Net 从 1 卡增到 4 卡反而变慢；最新 16 卡训练同时增大了 batch。不能把“更多卡”或“更大 batch”各自认定为质量收益。较早 JiT 的 lr 8e−4 两条分支在最终检查点明显退化，未确认原因。</p><p>后期验证 loss 仍下降，但 KID 改善变小；JiT 的最低 v-loss 也没有对应最低 KID。现有结果不证明已收敛，更不能用 loss 单独挑模型。</p></div>
<div class="card"><h3>下一步优先验证</h3><ol><li>增加生成样本数并复跑独立训练种子，确认 EMA、学习率的小差异。</li><li>在最新权重上复测适度区间引导，同时检查未筛选图与采样成本。</li><li>以相同看图量做 JiT patch 2 / 4、先大后小 batch 日程对照，逐项改变。</li></ol><p class="caption">这些是待验证方向，本次更新报告没有启动新训练；继续保持 RGB64。</p></div>
</div>
<p class="caption">统一质量协议：2,048 张真实参考、512 张生成图、50 个大小为 256 的 KID 子集；± 是子集标准差，不是置信区间。只有一个训练种子。参考项目 diffusion_demo2 的 48×48 数据与 clean-FID 协议不同，其 FID 数字不与这里的 KID 排名。</p>
</section>'''
