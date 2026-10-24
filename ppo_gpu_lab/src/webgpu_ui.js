(()=>{
 'use strict';const $=id=>document.getElementById(id),data=JSON.parse($('webgpu-data').textContent);
 let worker=null,state='idle',iteration=0,target=1200,session=null,current=null,result=null,history=[],validation=null,benchmark=null;
 const statusNames={idle:'未初始化',compiling:'连接 GPU / 编译中',ready:'可以开始训练',training:'正在训练',pausing:'本轮结束后暂停',paused:'已暂停 · 状态已保留',evaluating:'独立测试中',complete:'训练完成',validating:'数值校验中',benchmarking:'速度测量中',error:'发生错误'};
 function settings(){return{seed:+$('trainSeed').value,total:+$('trainBudget').value,task:$('trainTask').value};}
 function setState(s){state=s;$('trainStatus').textContent=statusNames[s]||s;const busy=['compiling','training','pausing','evaluating','validating','benchmarking'].includes(s);
  $('trainInit').disabled=busy;$('trainStart').disabled=busy||!worker||s==='error';$('trainPause').disabled=s!=='training';
  $('trainStart').textContent=s==='paused'?'继续训练':s==='complete'?`再训练 ${$('trainBudget').value} 轮`:'开始训练';
  for(const id of['trainTask','trainSeed','trainBudget'])$(id).disabled=busy;
  $('trainValidate').disabled=$('trainBench').disabled=busy||!worker||s==='error';$('trainUse').disabled=!current;$('trainExport').disabled=!current;
 }
 function plot(){const w=680,h=180,l=34,r=12,t=15,b=28,sx=x=>l+x/Math.max(target,1)*(w-l-r),sy=y=>h-b-y*(h-t-b);let a=[];
  for(const v of[0,.5,1])a.push(`<path d="M${l} ${sy(v)}H${w-r}" stroke="#d4decf"/><text x="${l-7}" y="${sy(v)+4}" text-anchor="end" font-size="10" fill="#6b7d6d">${v*100}%</text>`);
  for(const v of[0,target/2,target])a.push(`<text x="${sx(v)}" y="${h-7}" text-anchor="middle" font-size="10" fill="#6b7d6d">${Math.round(v)} 轮</text>`);
  if(history.length){a.push(`<path d="${history.map((p,i)=>(i?'L':'M')+sx(p.iteration)+' '+sy(p.evaluation.successRate)).join(' ')}" stroke="#26784e" stroke-width="2" fill="none"/>`);for(const p of history)a.push(`<circle cx="${sx(p.iteration)}" cy="${sy(p.evaluation.successRate)}" r="3" fill="#26784e"><title>${p.iteration} 轮：${p.evaluation.success}/${p.evaluation.count}</title></circle>`);}
  $('trainChart').innerHTML=a.join('');
 }
 function progress(m){iteration=m.iteration;target=m.target||target;$('trainIteration').textContent=`${iteration} / ${target} 轮`;$('trainSamples').textContent=(iteration*16384).toLocaleString('zh-CN')+' 个环境步';$('trainProgress').max=target;$('trainProgress').value=iteration;
  if(m.trainingMs!==undefined){$('trainTime').textContent=(m.trainingMs/1000).toFixed(2)+' s';if(m.trainingMs>0)$('trainSps').textContent=((iteration*16384)/(m.trainingMs/1000)/10000).toFixed(1)+' 万/s';}
  if(m.history){history=m.history;plot();const last=history.at(-1);if(last)$('trainSuccess').textContent=(last.evaluation.successRate*100).toFixed(1)+'%';}
  if(m.current){current=m.current;$('trainUse').disabled=false;$('trainExport').disabled=false;}
 }
 function useCurrent(label){if(!current)return;window.dispatchEvent(new CustomEvent('ppo-policy',{detail:{policy:current.policy,force:current.config.force,label:label||`WebGPU PPO · 第 ${current.iteration} 轮`}}));}
 function diagnostics(){let lines=[];if(validation){lines.push('数值校验：'+(validation.passed?'PASS':'FAIL'));for(const[k,v]of Object.entries(validation.checks))lines.push(k+': '+v.toExponential(3));}
  if(benchmark){lines.push('\n原任务速度基准 · 5 轮预热 + 30 轮测量',`完整训练中位数 ${benchmark.medianWallMs.toFixed(2)} ms / 轮 · ${(benchmark.sps/10000).toFixed(1)} 万环境步/s`,`P10–P90 ${benchmark.p10WallMs.toFixed(2)}–${benchmark.p90WallMs.toFixed(2)} ms · GPU 内核 ${benchmark.medianGpuMs===null?"计时器不可用":benchmark.medianGpuMs.toFixed(2)+" ms"}`);}
  $('trainDiagnostics').textContent=lines.join('\n')||'尚未运行本次校验 / 基准。';
 }
 function outcome(r){const rows=[['小扰动 ±2°','angle2'],['中扰动 ±5°','angle5'],['拖拽姿态 ±10°','angle10'],['±2° + 三次速度冲击','pushes']];$('trainOutcome').hidden=false;$('trainOutcome').innerHTML=`<b>已接入第 ${r.best.iteration} 轮的验证最佳快照，可在下方直接拖拽。</b><p>累计训练 ${(r.training.trainingMs/1000).toFixed(2)} s；${r.completedIterations.toLocaleString()} 轮。以下是独立 Float64 模拟器的留出测试，训练前后都使用 ${r.heldOut.force} N 电机。</p><table><thead><tr><th>测试范围</th><th>训练前</th><th>训练后</th></tr></thead><tbody>${rows.map(([label,k])=>`<tr><td>${label}</td><td>${r.baseline[k].success} / ${r.baseline[k].count}</td><td>${r.heldOut[k].success} / ${r.heldOut[k].count}</td></tr>`).join('')}</tbody></table><p class="train-small">成功要求 10 秒内未越界，且最后 1 秒连续稳定。训练与测速结果取决于本机 GPU、浏览器和后台负载。训练时间不含编译、评估和暂停。</p>`;}
 function raw(o){$('trainRaw').value=JSON.stringify(o,null,2);}
 function connect(autoStart=false){const s=settings();if(!Number.isInteger(s.seed)||s.seed<1||s.seed>999999||!Number.isInteger(s.total)||s.total<50||s.total>5000){$('trainError').hidden=false;$('trainError').textContent='请输入 1–999999 的整数种子，和 50–5000 的整数训练轮数。';return;}
  worker?.terminate();const url=URL.createObjectURL(new Blob([$('webgpu-worker-source').textContent],{type:'text/javascript'}));worker=new Worker(url);URL.revokeObjectURL(url);session=s;target=s.total;iteration=0;current=null;history=[];result=null;$('trainOutcome').hidden=true;$('trainSaved').hidden=true;$('trainError').hidden=true;$('trainSps').textContent=$('trainTime').textContent=$('trainSuccess').textContent='—';progress({iteration:0,target,history:[]});setState('compiling');
  worker.onerror=e=>{$('trainError').hidden=false;$('trainError').textContent=e.message;setState('error');};
  worker.onmessage=({data:m})=>{
   if(m.type==='state')setState(m.state);
   else if(m.type==='ready'){$('gpuDevice').textContent=`${m.device.vendor||'GPU'} / ${m.device.architecture||'WebGPU'} · ${(m.device.allocatedBytes/1048576).toFixed(2)} MiB · Worker · 编译 ${(m.device.setupMs/1000).toFixed(2)} s`;setState('ready');if(autoStart)worker.postMessage({type:'train',target});}
   else if(m.type==='progress')progress(m);
   else if(m.type==='paused'){current=m.current;progress(m);setState('paused');raw({paused:true,current});}
   else if(m.type==='complete'){result=m.result;current=result.best;progress({iteration:result.completedIterations,target,history:result.history,trainingMs:result.training.trainingMs});setState('complete');outcome(result);useCurrent();raw(result);}
   else if(m.type==='validation'){validation=m.result;diagnostics();raw(validation);setState(result?'complete':iteration?'paused':'ready');if(!validation.passed){$('trainError').hidden=false;$('trainError').textContent='数值校验未通过，请查看误差。';}}
   else if(m.type==='benchmark'){benchmark=m.result;diagnostics();raw(benchmark);setState(result?'complete':iteration?'paused':'ready');}
   else if(m.type==='snapshot'){current=m.current;useCurrent();}
   else if(m.type==='error'){$('trainError').hidden=false;$('trainError').textContent=m.message+'\n如果 WebGPU 不可用，请通过 localhost 在支持 WebGPU 的浏览器打开；已有策略仍可在下方测试。';setState('error');}
  };
  worker.postMessage({type:'init',initial:data.initial,fixture:data.fixture,target,config:{seed:s.seed,force:s.task==='robust'?40:10,robust:s.task==='robust'}});
 }
 $('trainInit').onclick=()=>connect();$('trainStart').onclick=()=>{const s=settings();if(!Number.isInteger(s.total)||s.total<50||s.total>5000){$('trainError').hidden=false;$('trainError').textContent='训练轮数应为 50–5000 的整数。';return;}if(!session||s.seed!==session.seed||s.task!==session.task){connect(true);return;}if(state==='complete')target=iteration+s.total;else if(state==='ready')target=s.total;setState('training');worker.postMessage({type:'train',target});};
 $('trainPause').onclick=()=>{setState('pausing');worker.postMessage({type:'pause'});};
 $('trainUse').onclick=()=>{useCurrent();$('interactive').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});};
 $('trainValidate').onclick=()=>{setState('validating');worker.postMessage({type:'validate'});};$('trainBench').onclick=()=>{setState('benchmarking');worker.postMessage({type:'benchmark'});};
 $('trainExport').onclick=()=>{const contents=result?{...result,validation,benchmark}:{current,validation,benchmark};const url=URL.createObjectURL(new Blob([JSON.stringify(contents,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='ppo_webgpu_seed'+(session?.seed||'sample')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
 if(data.recorded){current=data.recorded.best;useCurrent('已实测 WebGPU PPO · 第 '+current.iteration+' 轮');$('trainUse').disabled=false;$('trainExport').disabled=false;$('trainSaved').hidden=false;$('trainSaved').textContent=`下方已载入本机实测快照（seed 71，第 ${current.iteration} 轮），可以立即拖拽。上方按钮会新建训练，不需要等待已有快照重新训练。`;}
 window.addEventListener('pagehide',()=>worker?.terminate());plot();setState('idle');
})();
