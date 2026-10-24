"""Current RGB continuation section for the local experiment report."""

HTML = '''<style>#unet-continuation select{width:100%;max-width:100%}</style><section id="unet-continuation"><div class="num">HISTORICAL CONTROL / RGB U-NET / 5k → 40k</div>
<h2>先把 U-Net 训够，再比较架构</h2><p>U-Net 和 JiT 都直接处理 3×64×64 的带噪 RGB，并预测干净图像；两者均没有 VAE。pixel / latent 是表示空间的选择，U-Net / Transformer 是骨干结构的选择，不能混为同一项改进。</p><p class="callout" id="unetConclusion"></p>
<div class="card"><img class="gallery" src="results/rgb_exposure_summary.svg" alt="U-Net续训与两种JiT按图片曝光量对齐的验证损失和KID曲线"><p class="caption">横轴是图片张次，非不同图片数。左图：相同 512 张验证图、噪声与时间点，FP32 评估；右图：生成质量 KID，误差线为子集标准差，非置信区间。虚线为 16 万、64 万、128 万张次。</p></div>
<div class="card" style="margin-top:24px"><div class="cols"><div><label for="unetLeft">左侧模型</label><select id="unetLeft" aria-label="U-Net实验左侧模型"></select><img id="unetLeftImage" class="gallery" style="margin-top:12px" alt="左侧模型前64张固定未筛选RGB样本"></div><div><label for="unetRight">右侧模型</label><select id="unetRight" aria-label="U-Net实验右侧模型"></select><img id="unetRightImage" class="gallery" style="margin-top:12px" alt="右侧模型前64张固定未筛选RGB样本"></div></div><p class="caption">各模型相同初始噪声，Heun 50 区间 / 99 NFE；每组生成 512 张，展示前 64 张。没有筛图、颜色校正或中间裁剪。</p>
<div class="scroll"><table><thead><tr><th>模型 / 步数</th><th>图片张次</th><th>batch</th><th>墙钟 / PPU 分钟</th><th>EMA v-loss ↓</th><th>KID ×1,000 ↓</th></tr></thead><tbody id="unetComparisonRows"></tbody></table></div>
<p class="caption">KID：2,048 张真实验证图、512 张生成图，50 个子集 × 256 张，± 为子集标准差，并非置信区间。20k 小 JiT 与 20k U-Net 保持相同 batch、优化器与 EMA 配方；40k U-Net 与 10k 大 JiT 只对齐图片张次，大 JiT 的 batch、EMA 和容量均有变化。单训练种子，无多种子显著性结论。</p></div>
<div class="card" style="margin-top:24px"><div class="bar"><h3>U-Net 固定噪声训练过程</h3><span id="unetStageLabel"></span></div><input id="unetStage" type="range" min="0" value="0" aria-label="U-Net续训检查点"><img id="unetStageImage" class="gallery" style="max-width:650px;margin:auto" alt="U-Net续训固定16张RGB预览"><p class="caption" id="unetProgress"></p></div>
<p class="small" id="unetChecks"></p><details><summary>续训与卡数选择：保持 batch 32</summary><p>从原 5k 完整断点恢复模型、EMA、AdamW、GradScaler 和 CPU / PPU / NumPy 随机数状态，原 5k 目录保留。继续使用原训练脚本、RGB、lr=2e−4、随机水平翻转与单卡，不重置优化器或再次预热。</p><p id="unetScaling"></p><p class="caption">吞吐基准使用相同 5k 权重与驻留显存的 RGB 输入，含前反向、梯度同步、裁剪、AdamW 和 EMA；不含 I/O、验证和存盘。40 步预热后测 5 组 × 20 步。此结果针对小 U-Net 和固定 batch 32。</p></details>
<div class="linkrow small"><a href="results/rgb_flow_unet40k/training.json">完整续训曲线</a><a href="results/rgb_unet_continuation_evaluation/evaluation.json" id="unetEvalLink">采样与数值检查</a><a href="results/rgb_unet_continuation_evaluation/kid.json" id="unetKidLink">KID 原始数据</a><a href="results/unet_scaling_1ppu/benchmark.json">1 卡基准</a><a href="results/unet_scaling_4ppu/benchmark.json">4 卡基准</a><a href="src/evaluate_unet_continuation.py">评估代码</a><a href="results/rgb_unet_environment.json">续训源码与权重哈希</a></div></section>'''

JS = r'''
const ut=D.unet_train, ue=D.unet_eval, uk=Object.fromEntries((D.unet_kid.results||[]).map(x=>[x.name,x]));
const uchoices=[];
for(const [key,name,label] of [['unet','flow_unet5k','U-Net 6.62M / 5k'],['jit5k','flow_jit5k','JiT 6.24M / 5k'],['jit','flow_jit20k','JiT 6.24M / 20k']]){
 const md=D.flow.models?.[key]?.training;
 if(md)uchoices.push({name,label,path:`results/rgb_flow_evaluation/${name}_64.png`,md,batch:32,ppus:1,kid:smallK[name]});
}
for(const r of ue.rows||[])uchoices.push({name:r.name,label:`U-Net 6.62M / ${r.step/1000}k`,path:`results/rgb_unet_continuation_evaluation/${r.name}_64.png`,md:r.training,batch:32,ppus:1,kid:uk[r.name]});
for(const [i,e] of D.round2.evals.entries())for(const r of e.rows||[])if(r.run==='small_jit40k')uchoices.push({name:'flow_jit40k',label:'JiT 6.24M / 40k',path:`results/rgb_round2_evaluation_${i}/${r.name}_64.png`,md:r.training,batch:32,ppus:1,kid:r});
for(const r of D.scaled_eval.rows||[])uchoices.push({name:r.name,label:`JiT 32.63M / ${r.step.toLocaleString()}步`,path:`results/rgb_scaled_evaluation/${r.name}_64.png`,md:r.training,batch:128,ppus:4,kid:sk[r.name]});
uchoices.sort((a,b)=>a.md.step*a.batch-b.md.step*b.batch);
for(const side of ['Left','Right']){
 const sel=$('unet'+side);sel.innerHTML=uchoices.map((x,i)=>`<option value="${i}">${x.label} · ${fmt(x.md.step*x.batch/1000,0)}k张次</option>`).join('');
 const preferred=side==='Left'?(uk.flow_unet40k?'flow_unet40k':uk.flow_unet20k?'flow_unet20k':'flow_unet5k'):(uk.flow_unet40k?(uchoices.some(x=>x.name==='flow_jit40k')?'flow_jit40k':'scaled_jit10000'):uk.flow_unet20k?'flow_jit20k':'flow_jit5k');
 sel.value=Math.max(0,uchoices.findIndex(x=>x.name===preferred));
 function show(){$('unet'+side+'Image').src=uchoices[Number(sel.value)].path;}
 sel.addEventListener('change',show);if(uchoices.length)show();
}
$('unetComparisonRows').innerHTML=uchoices.map(x=>`<tr${x.name.startsWith('flow_unet')?' style="background:#f7efe3"':''}><td>${x.label}</td><td>${(x.md.step*x.batch).toLocaleString()}</td><td>${x.batch}</td><td>${fmt(x.md.elapsed_seconds/60)} / ${fmt(x.md.elapsed_seconds*x.ppus/60)}</td><td>${fmt(x.md.curve.at(-1).validation_ema.mse,5)}</td><td>${x.kid?fmt(x.kid.kid_mean*1000,2)+' ± '+fmt(x.kid.subset_std*1000,2):'待评估'}</td></tr>`).join('');
if(ut.step){
 const old=D.flow.models.unet.training.curve.at(-1).validation_ema.mse, latest=ut.curve.at(-1).validation_ema.mse;
 let text=`U-Net 从 5k 续到 ${ut.step.toLocaleString()} 步，验证 v-loss 从 ${fmt(old,5)} 降到 ${fmt(latest,5)}，下降 ${fmt((1-latest/old)*100,1)}%。${ut.status==='complete'?'40k 训练已完成。':'正在继续到 40k，RGB 配方保持不变。'}`;
 if(uk.flow_unet20k)text+=`同为 64 万张次，U-Net / 小 JiT 的 KID×1,000 为 ${fmt(uk.flow_unet20k.kid_mean*1000,2)} / ${fmt(smallK.flow_jit20k.kid_mean*1000,2)}。`;
 if(uk.flow_unet40k)text+=`同为 128 万张次，U-Net / 大 JiT 为 ${fmt(uk.flow_unet40k.kid_mean*1000,2)} / ${fmt(sk.scaled_jit10000.kid_mean*1000,2)}。`;
 const j40=uchoices.find(x=>x.name==='flow_jit40k');if(j40)text+=`补充的小 JiT40k 为 ${fmt(j40.kid.kid_mean*1000,2)}。`;
 $('unetConclusion').textContent=text;$('conclusion').textContent=text;
 $('status').textContent=`RGB U-Net / 1 PPU · ${ut.step.toLocaleString()} / 40,000`;
 $('steps').textContent=ut.step.toLocaleString();$('duration').textContent=fmt(ut.elapsed_seconds/60,1)+' 分';$('mse').textContent=fmt(latest,4);$('latency').textContent=uk.flow_unet40k?fmt(uk.flow_unet40k.kid_mean*1000,2):'待 40k';
 document.querySelectorAll('.metrics .metric span').forEach((e,i)=>e.textContent=['U-Net · batch 32 / 目标 40k','U-Net 累计墙钟 · 包含原 5k','U-Net 当前 EMA v-loss ↓','U-Net 40k RGB KID ×1,000 ↓'][i]);
 $('unetProgress').textContent=`${ut.training_image_draws.toLocaleString()} 图片张次；成功更新 ${ut.successful_updates.toLocaleString()}，AMP 跳过 ${ut.skipped}；累计墙钟包含原 5k 以及训练内验证、预览和存盘，不含进程启动与离线质量评估。`;
}
const us=D.unet_stages;
if(us.length){$('unetStage').max=us.length-1;$('unetStage').value=us.length-1;function showUnet(){const step=us[Number($('unetStage').value)];$('unetStageLabel').textContent=step.toLocaleString()+' 步';$('unetStageImage').src=`results/${step<=5000?'rgb_flow_unet5k':'rgb_flow_unet40k'}/ema_step${step}.png`;}$('unetStage').addEventListener('input',showUnet);showUnet();}
const u1=D.unet_scaling.find(x=>x.world_size===1),u4=D.unet_scaling.find(x=>x.world_size===4);
if(u1&&u4)$('unetScaling').textContent=`固定总 batch 32：1 卡 ${fmt(u1.median_ms,2)} ms/步（${fmt(u1.images_per_second,0)} 图/秒），4 卡 ${fmt(u4.median_ms,2)} ms/步（${fmt(u4.images_per_second,0)} 图/秒），4 卡步耗时增加 ${fmt((u4.median_ms/u1.median_ms-1)*100,1)}%，因此该阶段固定 batch32 的续训采用 1 卡；后续 16 卡分支将 global batch 增至 512。分布式与单卡 FP32 全局梯度相对 L2 误差 ${u4.numerical.gradient_relative_l2.toExponential(2)}；两组计时均无跳步。`;
const uc=ue.training_stream_checks_at20k;
$('unetChecks').textContent=(uc?`20k 训练随机流检查：U-Net 与小 JiT 的 NumPy / PPU RNG 状态${uc.numpy_rng_equal&&uc.cuda_rng_equal?'均相同':'存在差异'}。`:'20k 随机流检查待运行。')+(ue.rows||[]).map(x=>`${x.step/1000}k 的 FP16 Graph / 普通 FP32 最终图像 RMSE：${x.amp_graph_vs_fp32_eager_rmse.toExponential(2)}。`).join('');
if(!(ue.rows||[]).length)$('unetEvalLink').style.display='none';
if(!(D.unet_kid.results||[]).length)$('unetKidLink').style.display='none';
'''
