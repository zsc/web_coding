let hybridTarget=Uint8Array.from(HYBRID_TARGET.pixels),hybridEngine=null,hybridLatest=null,hybridBusy=false,hybridStop=false;
const hybridHistory=[];
paint('live-target',hybridTarget,320,128);paint('live-image',new Uint8Array(40960),320,128);
function hybridBusyState(busy){hybridBusy=busy;for(const id of ['demo-start','demo-benchmark','demo-text','demo-N','demo-steps','demo-adam','demo-chains','demo-precision','demo-file'])el(id).disabled=busy;el('demo-stop').disabled=!busy;}
function hybridSettings(){for(const id of ['demo-N','demo-steps','demo-adam'])if(!el(id).reportValidity())return null;return {W:320,H:128,N:Number(el('demo-N').value),C:Number(el('demo-chains').value),mcmc:Number(el('demo-steps').value),adam:Number(el('demo-adam').value),mode:el('demo-precision').value,polish:1000};}
function hybridRefresh(r){paint('live-image',r.image,320,128);el('live-iter').textContent=r.adam_done?`Adam ${r.adam_done.toLocaleString()} / ${r.adam.toLocaleString()}`:`ROI ${r.mcmc_done.toLocaleString()} / ${r.mcmc.toLocaleString()}`;el('live-loss').textContent=(r.loss??r.cached_loss).toFixed(4)+(r.loss===undefined?' · GPU':' · FP64');el('live-gpu').textContent=r.gpu_ms===null?'不可用':(r.gpu_ms/1000).toFixed(2)+' s';el('live-time').textContent=((r.total_ms??r.fit_wall_ms)/1000).toFixed(2)+' s';el('live-memory').textContent=(r.workspace_bytes/1048576).toFixed(1)+' MiB';}
function hybridValidate(r){return new Promise((resolve,reject)=>{const source=validateHybrid.toString()+';self.onmessage=({data})=>{try{self.postMessage(validateHybrid(data));}catch(e){self.postMessage({error:e.message});}}';const url=URL.createObjectURL(new Blob([source],{type:'text/javascript'})),worker=new Worker(url);const done=()=>{worker.terminate();URL.revokeObjectURL(url);};worker.onmessage=({data})=>{done();data.error?reject(Error(data.error)):resolve(data);};worker.onerror=e=>{done();reject(Error(e.message));};worker.postMessage({...r,target:hybridTarget});});}
async function hybridOne(settings,label=''){
 const start=performance.now(),cold=!hybridEngine;el('live-status').textContent=label+(cold?'正在检测 GPU / 编译管线…':'正在初始化 GPU 状态…');
 if(!hybridEngine)hybridEngine=await HybridWebGPU.create();
 el('hybrid-device').textContent=`${hybridEngine.adapter.info?.vendor||'GPU'} / ${hybridEngine.adapter.info?.architecture||''} · shader-f16 ${hybridEngine.half?'✓':'×'} · subgroups ${hybridEngine.subgroups?'✓':'×'} · timestamp ${hybridEngine.timestamp?'✓':'×'}`;
 if(hybridStop){el('live-status').textContent='已取消本次运行。';return null;}
 const result=await hybridEngine.run(hybridTarget,settings,r=>{hybridRefresh(r);el('live-status').textContent=label+(hybridStop?'停止中…':r.adam_done?'正在并行细化全部高斯…':'正在 ROI 局部搜索…');});
 el('live-status').textContent=label+'正在用 Float64 全图重渲染验收…';el('demo-stop').disabled=true;
 const validation=await hybridValidate(result);
 const final={...result,...validation,cold_pipeline:cold,compile_ms:cold?hybridEngine.compile_ms:0,total_ms:performance.now()-start,at:new Date().toISOString(),target:el('demo-text').value};
 hybridLatest=final;hybridRefresh(final);el('demo-export').disabled=false;
 const {image,gaussians,...metadata}=final;hybridHistory.push(metadata);
 el('hybrid-evidence').textContent=JSON.stringify(hybridHistory,null,2);el('hybrid-result').textContent=JSON.stringify({...final,image:Array.from(image)},null,2);
 el('live-status').textContent=`${label}${result.cancelled?'已停止，当前结果已验证':'完成，最终结果已验证'} · ${(final.total_ms/1000).toFixed(2)} s · MSE ${final.loss.toFixed(4)}${cold?' · 含管线编译':' · 复用管线'}`;
 return final;
}
async function hybridRun(benchmark){if(hybridBusy)return;const settings=hybridSettings();if(!settings)return;hybridStop=false;hybridBusyState(true);el('demo-export').disabled=true;
 try{if(benchmark){for(let round=0;round<3&&!hybridStop;round++){for(const mode of [...['fp32','fp16','fp16-cache'].slice(round),...['fp32','fp16','fp16-cache'].slice(0,round)]){if(hybridStop)break;el('demo-stop').disabled=false;await hybridOne({...settings,mode},`第 ${round+1}/3 轮 · ${mode} · `);}}}else await hybridOne(settings);}
 catch(error){el('live-status').textContent='运行失败：'+error.message;if(hybridEngine?.lost){hybridEngine.destroy();hybridEngine=null;}}
 finally{hybridBusyState(false);}
}
el('demo-start').addEventListener('click',()=>hybridRun(false));el('demo-benchmark').addEventListener('click',()=>hybridRun(true));
el('demo-stop').addEventListener('click',()=>{hybridStop=true;hybridEngine?.cancel();el('demo-stop').disabled=true;el('live-status').textContent='正在完成已提交的小批次，随后保留并验证结果…';});
function hybridReset(target){hybridTarget=target;hybridLatest=null;for(const id of ['live-iter','live-loss','live-gpu','live-time','live-memory'])el(id).textContent='—';paint('live-target',target,320,128);paint('live-image',new Uint8Array(40960),320,128);el('demo-export').disabled=true;el('live-status').textContent='目标已更新。点击初始化并运行 WebGPU。';}
el('demo-text').addEventListener('change',()=>{el('demo-file').value='';if(el('demo-text').value==='周舒畅'){hybridReset(Uint8Array.from(HYBRID_TARGET.pixels));return;}const c=document.createElement('canvas');c.width=320;c.height=128;const x=c.getContext('2d');x.fillStyle='black';x.fillRect(0,0,320,128);x.fillStyle='white';x.font='96px sans-serif';x.textAlign='center';x.textBaseline='middle';x.fillText(el('demo-text').value,160,64,308);const a=x.getImageData(0,0,320,128).data;hybridReset(Uint8Array.from({length:40960},(_,i)=>a[i*4]));});
el('demo-file').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;hybridBusyState(true);el('demo-stop').disabled=true;try{const bm=await createImageBitmap(file),c=document.createElement('canvas');c.width=320;c.height=128;const x=c.getContext('2d');x.fillStyle='black';x.fillRect(0,0,320,128);x.drawImage(bm,0,0,320,128);bm.close();const a=x.getImageData(0,0,320,128).data;el('demo-text').value='';hybridReset(Uint8Array.from({length:40960},(_,i)=>Math.round(.2126*a[4*i]+.7152*a[4*i+1]+.0722*a[4*i+2])));}catch(error){el('live-status').textContent='无法读取图片：'+error.message;}finally{hybridBusyState(false);}});
el('target-export').addEventListener('click',()=>download(new Blob(['P5\n320 128\n255\n',hybridTarget],{type:'image/x-portable-graymap'}),'target.pgm'));
el('demo-export').addEventListener('click',()=>{if(!hybridLatest)return;download(new Blob([JSON.stringify({...hybridLatest,image:Array.from(hybridLatest.image)},null,2)],{type:'application/json'}),'webgpu-fit.json');const c=document.createElement('canvas');paint(c,hybridLatest.image,320,128);c.toBlob(blob=>download(blob,'webgpu-fit.png'));});
