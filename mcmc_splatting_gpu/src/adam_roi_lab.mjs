import {HybridWebGPU,validateHybrid} from './hybrid_webgpu.mjs';
import {validateHybridParallel} from './parallel_validate.mjs';
const $=id=>document.getElementById(id),targetData=await (await fetch('assets/zhou_target.json')).json(),target=Uint8Array.from(targetData.pixels);
function paint(id,pixels){const ctx=$(id).getContext('2d'),im=ctx.createImageData(320,128);for(let p=0;p<pixels.length;p++){im.data.set([pixels[p],pixels[p],pixels[p],255],p*4);}ctx.putImageData(im,0,0);}
paint('target',target);let busy=false;
const variants=[{name:'6σ / 单 Worker',sigma:6,workers:1},{name:'4.5σ / 单 Worker',sigma:4.5,workers:1},{name:'4.5σ / 4 Workers',sigma:4.5,workers:4},{name:'4σ / 单 Worker',sigma:4,workers:1}];
async function one(v,phase,repeat,study){
 const start=performance.now();$('status').textContent=`${phase} ${repeat+1} · ${v.name} · 编译与计算中…`;const engine=await HybridWebGPU.create({roiSigma:v.sigma});
 try{
  const r=await engine.run(target,{W:320,H:128,N:1600,C:8,mcmc:0,adam:8000,mode:'fp32'},()=>{});
  const validation=await validateHybridParallel({...r,target},v.workers);const total=performance.now()-start;
  // Independent single-worker-style reference outside the measured interval, once per setting.
  let check=null;if(!study.representatives[v.name]){const ref=validateHybrid({...r,target});let different=0;for(let p=0;p<ref.image.length;p++)if(ref.image[p]!==validation.image[p])different++;if(different||Math.abs(ref.loss-validation.loss)>1e-9)throw Error('并行验收与串行参考不一致');check={different_pixels:different,serial_mse:ref.loss,passed:true};}
  const metadata={...v,phase,repeat,total_ms:total,gpu_ms:r.gpu_ms,compile_ms:engine.compile_ms,validation_ms:validation.validation_ms,loss:validation.loss,cached_loss:r.cached_loss,render_mae:validation.render_mae,adapter:r.adapter,features:r.features,parallel_check:check};study.runs.push(metadata);
  if(phase==='repeat')study.representatives[v.name]={...r,...validation,image:Array.from(validation.image),target:Array.from(target),...metadata};
  paint('fit',validation.image);$('metrics').textContent=`${(total/1000).toFixed(3)} s · MSE ${validation.loss.toFixed(4)}`;$('device').textContent=`${r.adapter.vendor} / ${r.adapter.architecture} · shader-f16 ${r.features.half} · subgroups ${r.features.subgroups}`;
  const tr=document.createElement('tr');for(const txt of [v.name+(phase==='warmup'?' · 预热':''),(total/1000).toFixed(3)+' s',(r.gpu_ms/1000).toFixed(3)+' s',(validation.validation_ms/1000).toFixed(3)+' s',validation.loss.toFixed(4)]){const td=document.createElement('td');td.textContent=txt;tr.append(td);}$('rows').append(tr);publish(study);
 }finally{engine.destroy();}
}
function publish(study){const {representatives,...metadata}=study;$('bench-summary').textContent=JSON.stringify(metadata,null,2);$('bench-data').textContent=JSON.stringify(study);}
async function start(benchmark){if(busy)return;busy=true;for(const id of ['run','bench','sigma','workers'])$(id).disabled=true;$('rows').textContent='';const study={complete:false,method:'Fresh engine/pipelines each run. Total includes compilation, optimization, readback, worker creation and full Float64 validation; excludes page load and separate serial cross-check. Fixed seed42+1009*c. First uncounted warmup; 3 rotating rounds.',runs:[],representatives:{},sources:{}};
 try{
  for(const path of ['src/hybrid_webgpu.mjs','src/parallel_validate.mjs','src/adam_roi_lab.mjs','assets/zhou_target.json']){const bytes=await(await fetch(path)).arrayBuffer();study.sources[path]=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');}
  if(benchmark){await one(variants[0],'warmup',0,study);for(let repeat=0;repeat<3;repeat++)for(let i=0;i<variants.length;i++)await one(variants[(i+repeat)%variants.length],'repeat',repeat,study);}
  else await one({name:`${$('sigma').value}σ / ${$('workers').value} Workers`,sigma:Number($('sigma').value),workers:Number($('workers').value)},'repeat',0,study);
  study.complete=true;publish(study);$('status').textContent='完成：所有结果已通过完整 Float64 验收。';
 }catch(e){study.error=e.message;publish(study);$('status').textContent='失败：'+e.message;}
 finally{busy=false;for(const id of ['run','bench','sigma','workers'])$(id).disabled=false;}
}
$('run').addEventListener('click',()=>start(false));$('bench').addEventListener('click',()=>start(true));
