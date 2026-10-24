import {validateHybrid} from './hybrid_webgpu.mjs';
// Disjoint row partitions preserve the full Gaussian sum in Float64 for each pixel.
export async function validateHybridParallel(data,threads=1){
 if(!Number.isInteger(threads)||threads<1||threads>8)throw Error('validation threads must be 1..8');
 const start=performance.now(),workers=[],source=validateHybrid.toString()+';self.onmessage=({data})=>{try{const r=validateHybrid(data);self.postMessage(r,[r.image.buffer]);}catch(e){self.postMessage({error:e.message});}}',url=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
 try{
  const chunks=await Promise.all(Array.from({length:threads},(_,i)=>new Promise((resolve,reject)=>{
   const y0=Math.floor(data.H*i/threads),y1=Math.floor(data.H*(i+1)/threads),g=data.gaussians.slice();for(let j=1;j<g.length;j+=12)g[j]-=y0;
   const worker=new Worker(url);workers.push(worker);worker.onerror=e=>reject(Error(e.message));worker.onmessage=({data:r})=>r.error?reject(Error(r.error)):resolve({y0,y1,...r});
   worker.postMessage({...data,H:y1-y0,gaussians:g,target:data.target.slice(y0*data.W,y1*data.W),image:data.image.slice(y0*data.W,y1*data.W)});
  })));
  const image=new Uint8Array(data.W*data.H);let sse=0,mae=0,max=0;
  for(const r of chunks){const pixels=(r.y1-r.y0)*data.W;image.set(r.image,r.y0*data.W);sse+=r.loss*pixels;mae+=r.render_mae*pixels;max=Math.max(max,r.render_max_error);}
  const loss=sse/image.length;return {image,loss,psnr:10*Math.log10(65025/loss),render_mae:mae/image.length,render_max_error:max,validation_ms:performance.now()-start,validation_threads:threads};
 }finally{for(const w of workers)w.terminate();URL.revokeObjectURL(url);}
}
