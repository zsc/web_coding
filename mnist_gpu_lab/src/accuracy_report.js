if(D.accuracy){
  const A=D.accuracy,F=A.final,first=F.seeds[0],second=F.seeds[1],oldErrors=errors.length;
  const baseline={predictions:D.predictions,probabilities:D.probabilities,confusion:D.confusion,checks:$('checkList').innerHTML,caption:$('explorerCaption').textContent,note:$('exportNote').textContent};
  $('accuracy').classList.remove('hidden');
  document.querySelectorAll('#accuracy a[href*="model.pt"],#accuracy a[href*="tfjs_accuracy"],#exportPrimary,#exportJson,#exportTFModel,#exportTFWeights').forEach(el=>el.remove());
  const newErrors=Math.round((1-first.test.accuracy)*10000);
  $('newAccuracy').textContent=pct(first.test.accuracy);$('repeatAccuracy').textContent=pct(second.test.accuracy);$('ensembleAccuracy').textContent=pct(F.ensemble.accuracy);
  $('heroAcc').parentElement.querySelector('small').textContent='原版速度基线的测试准确率';
  $('ensembleAccuracy').parentElement.nextElementSibling.textContent=`集成准确率 ${pct(F.ensemble.accuracy)}，${F.ensemble.accuracy<=Math.max(first.test.accuracy,second.test.accuracy)?'本次没有超过较好的单模型，因此优先用单模型。':'需要两次模型前向，属于额外推理成本。'}`;
  $('newErrors').textContent=`错误 ${oldErrors} → ${newErrors} 张，相对错误率减少 ${fmt((oldErrors-newErrors)/oldErrors*100,1)}%。原交付模型 ${pct(acc(R.cuda_graph))}。`;
  const labels={small_constant:'原宽度 · 固定学习率',small_cosine:'原宽度 + 余弦衰减',wide_cosine:'加宽 + 余弦衰减',wide_aug_cosine:'加宽 + 余弦衰减 + 增强'};
  $('accuracyAblations').innerHTML=A.search.map(r=>`<tr class="${r.config.name===F.selection.config.name?'selectedrow':''}"><td>${labels[r.config.name]} ${r.config.name===F.selection.config.name?'<span class="badge">验证集选中</span>':''}</td><td>${r.parameter_count.toLocaleString()}</td><td>${pct(r.validation.accuracy)}</td><td>${fmt(r.validation.loss,4)}</td><td>${time(r.train_seconds)}</td></tr>`).join('');
  $('accuracyCost').textContent=`最终选中 ${labels[F.selection.config.name]}，${F.selection.epochs} epochs，在全部 60,000 张上重训。seed 42 / 43 训练耗时 ${time(first.train_seconds)} / ${time(second.train_seconds)}，含图准备及训练日志为 ${time(first.training_pipeline_seconds)} / ${time(second.training_pipeline_seconds)}，均另计一次测试与导出。单模型参数 ${first.parameter_count.toLocaleString()}（原来的 ${fmt(first.parameter_count/R.cuda_graph.parameter_count)}×），训练期间 PyTorch 峰值 ${fmt(first.peak_allocated_mib,0)} MiB。GPU 共享负载下的本次实测，不作固定延迟承诺。`;
  $('testModel').insertAdjacentHTML('beforeend','<option value="improved">准确率升级 · seed 42</option>');
  function switchTestModel(){
    const improve=$('testModel').value==='improved',data=improve?A:baseline;
    D.predictions=data.predictions;D.probabilities=data.probabilities;D.confusion=data.confusion;
    errors=D.labels.map((v,i)=>v!==D.predictions[i]?i:-1).filter(i=>i>=0);drawConfusion();
    $('explorerCaption').textContent=improve?`准确率升级模型 / ${F.selection.epochs} epochs / seed 42，完整测试准确率 ${pct(first.test.accuracy)}。随机抽取真实测试图与保存的模型输出，不会重新训练。`:baseline.caption;
    $('checkList').innerHTML=improve?[`10,000 张测试样本：${10000-errors.length} 正确，${errors.length} 错分。`,'训练 / 验证分割不重叠，验证集每类 500 张。',`Graph 动态学习率检查：最大权重误差 ${Math.max(...A.correctness.changing_lr_weight_max_errors).toExponential(2)}。`,`学习率设为 0：参数变化 ${A.correctness.zero_lr_parameter_delta}。`, `完整 checkpoint 复读：预测一致率 ${pct(first.export_prediction_agreement)}。`,`TF.js 回读 ${A.tfjs.samples} 张：概率最大误差 ${A.tfjs.max_probability_absolute_error.toExponential(2)}。`].map(x=>`<li>${x}</li>`).join(''):baseline.checks;
    $('exportNote').textContent=improve?`新模型由验证集选方案，全 60k 重训 ${F.selection.epochs} 轮，使用预先指定的 seed 42；没有选择测试最优 epoch。权重与 TF.js 导出仅本地保留。`:baseline.note;
    if(img.complete&&img.naturalWidth)randomSamples();
  }
  $('testModel').onchange=switchTestModel;$('testModel').value='improved';switchTestModel();
}
