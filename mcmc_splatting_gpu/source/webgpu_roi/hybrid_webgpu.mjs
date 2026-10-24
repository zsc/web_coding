// ROI MCMC + sparse analytic Adam, ported from roi_fit.cu.
// Integer splat accumulator: 1/1024 intensity units; N<=10000 bounds the sum below u32 max.
export function hybridShader(half = false, subgroups = true, cached = false, roiSigma = 6) {
if(!Number.isFinite(roiSigma)||roiSigma<3||roiSigma>6)throw Error("ROI sigma must be between 3 and 6");
const reduce = subgroups ? `ra[tid]=subgroupAdd(ga);rb[tid]=subgroupAdd(gb);workgroupBarrier();
if(tid==0u){ga=vec4f(0);gb=vec2f(0);for(var t=0u;t<32u;t+=sgSize){ga+=ra[t];gb+=rb[t];}}`
: `ra[tid]=ga;rb[tid]=gb;workgroupBarrier();for(var d=16u;d>0u;d/=2u){if(tid<d){ra[tid]+=ra[tid+d];rb[tid]+=rb[tid+d];}workgroupBarrier();}ga=ra[0];gb=rb[0];`;
return `${half?'enable f16;':''}\n${subgroups?'enable subgroups;':''}
const ROI_SIGMA:f32=${roiSigma.toFixed(4)};
const ROI_LIMIT:f32=ROI_SIGMA*ROI_SIGMA;
struct G {x:f32,y:f32,sx:f32,sy:f32,angle:f32,a:f32,co:f32,si:f32,ix:f32,iy:f32,rx:f32,ry:f32};
struct Config {W:u32,H:u32,N:u32,C:u32,step:u32,total:u32,batch:u32,offset:u32,rate:f32,c1:f32,c2:f32,cacheStride:u32};
struct State {rng:u32,accepted:u32,sse:u32,iteration:u32};
@group(0) @binding(0) var<uniform> cfg:Config;
@group(0) @binding(1) var<storage,read_write> gs:array<G>;
@group(0) @binding(2) var<storage,read_write> sums:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> states:array<State>;
@group(0) @binding(4) var<storage,read> targetImage:array<f32>;
@group(0) @binding(5) var<storage,read_write> moments:array<vec2f>;
${cached?"@group(0) @binding(6) var<storage,read_write> expCache:array<f16>;":""}
var<workgroup> ra:array<vec4f,32>;
var<workgroup> rb:array<vec2f,32>;
var<workgroup> ri:array<u32,32>;
var<workgroup> oldg:G;var<workgroup> newg:G;
var<workgroup> rect:vec4i;var<workgroup> chosen:u32;var<workgroup> yes:u32;
var<workgroup> acceptU:f32;var<workgroup> temperature:f32;
fn pack(a:G)->G {var g=a;g.co=cos(-g.angle);g.si=sin(-g.angle);g.ix=1.0/g.sx;g.iy=1.0/g.sy;
g.rx=ROI_SIGMA*sqrt(g.sx*g.sx*g.co*g.co+g.sy*g.sy*g.si*g.si);g.ry=ROI_SIGMA*sqrt(g.sx*g.sx*g.si*g.si+g.sy*g.sy*g.co*g.co);return g;}
fn bounds(g:G)->vec4i {let lo=max(vec2i(0),vec2i(floor(vec2f(g.x-g.rx,g.y-g.ry))));let hi=min(vec2i(i32(cfg.W)-1,i32(cfg.H)-1),vec2i(ceil(vec2f(g.x+g.rx,g.y+g.ry))));return vec4i(lo,max(vec2i(0),hi-lo+1));}
fn exponent(q:f32)->f32 {return ${half?'f32(exp(f16(-0.5*q)))':'exp(-0.5*q)'};}
fn amplitude(a:f32,e:f32)->f32 {return ${half?'f32(f16(a)*f16(e))':'a*e'};}
fn value(g:G,x:i32,y:i32)->f32 {let dx=f32(x)-g.x;let dy=f32(y)-g.y;let u=(dx*g.co-dy*g.si)*g.ix;let v=(dx*g.si+dy*g.co)*g.iy;let q=u*u+v*v;if(q>ROI_LIMIT){return 0.0;}return amplitude(g.a,exponent(q));}
fn quantize(s:f32)->i32{return i32(clamp(floor(s+0.5),0.0,255.0));}
fn fixed(v:f32)->u32{return u32(floor(v*1024.0+0.5));}
fn next(r:u32)->u32 {var s=r;s^=s<<13u;s^=s>>17u;s^=s<<5u;return s;}
fn randomUnit(r:u32)->f32{return f32(r>>8u)*(1.0/16777216.0);}
@compute @workgroup_size(256)
fn clear(@builtin(global_invocation_id) id:vec3u){if(id.x<cfg.W*cfg.H*cfg.C){atomicStore(&sums[id.x],0u);}}
@compute @workgroup_size(32)
fn render(@builtin(local_invocation_index) tid:u32,@builtin(workgroup_id) bid:vec3u){
let g=gs[bid.y*cfg.N+bid.x];let b=bounds(g);let base=bid.y*cfg.W*cfg.H;
for(var q=i32(tid);q<b.z*b.w;q+=32){let x=b.x+q%b.z;let y=b.y+q/b.z;let dx=f32(x)-g.x;let dy=f32(y)-g.y;let u=(dx*g.co-dy*g.si)*g.ix;let v=(dx*g.si+dy*g.co)*g.iy;let qq=u*u+v*v;var e=0.0;if(qq<=ROI_LIMIT){e=exponent(qq);}
${cached?'if(u32(q)<cfg.cacheStride){expCache[(bid.y*cfg.N+bid.x)*cfg.cacheStride+u32(q)]=f16(e);}':''}
let contribution=fixed(amplitude(g.a,e));if(contribution!=0u){atomicAdd(&sums[base+u32(y)*cfg.W+u32(x)],contribution);}}
}
@compute @workgroup_size(32)
fn loss(@builtin(local_invocation_index) tid:u32,@builtin(workgroup_id) bid:vec3u){
let P=cfg.W*cfg.H;var s=0u;for(var p=tid;p<P;p+=32u){let d=quantize(f32(atomicLoad(&sums[bid.x*P+p]))/1024.0)-i32(targetImage[p]);s+=u32(d*d);}
ri[tid]=s;workgroupBarrier();for(var d=16u;d>0u;d/=2u){if(tid<d){ri[tid]+=ri[tid+d];}workgroupBarrier();}if(tid==0u){states[bid.x].sse=ri[0];}
}
@compute @workgroup_size(32)
fn mcmc(@builtin(local_invocation_index) tid:u32,@builtin(workgroup_id) bid:vec3u){
let c=bid.x;let P=cfg.W*cfg.H;let base=c*P;
for(var it=0u;it<cfg.batch;it++){
 if(tid==0u){var r=next(states[c].rng);chosen=min(u32(randomUnit(r)*f32(cfg.N)),cfg.N-1u);r=next(r);let k=min(u32(randomUnit(r)*6.0),5u);r=next(r);let delta=(randomUnit(r)-0.5)*2.0;r=next(r);acceptU=randomUnit(r);states[c].rng=r;
 let ratio=f32(cfg.offset+it)/f32(max(1u,cfg.total-1u));temperature=2.0*pow(0.0005/2.0,ratio);let factor=0.15+0.85*pow(0.0005/2.0,ratio*0.25);
 oldg=gs[c*cfg.N+chosen];var g=oldg;
 switch(k){case 0u:{g.x=clamp(g.x+delta*2.0*factor,0.0,f32(cfg.W));}case 1u:{g.y=clamp(g.y+delta*2.0*factor,0.0,f32(cfg.H));}case 2u:{g.sx=max(0.1,g.sx+delta*factor);}case 3u:{g.sy=max(0.1,g.sy+delta*factor);}case 4u:{g.angle+=delta*0.1*factor;}default:{g.a=clamp(g.a+delta*10.0*factor,0.0,255.0);}}
 newg=pack(g);let a=bounds(oldg);let b=bounds(newg);let lo=min(a.xy,b.xy);let hi=max(a.xy+a.zw,b.xy+b.zw);rect=vec4i(lo,hi-lo);
 }workgroupBarrier();var ds=0u;
 for(var q=i32(tid);q<rect.z*rect.w;q+=32){let x=rect.x+q%rect.z;let y=rect.y+q/rect.z;let p=base+u32(y)*cfg.W+u32(x);
 let delta=fixed(value(newg,x,y))-fixed(value(oldg,x,y));let old=atomicLoad(&sums[p]);let a=quantize(f32(old)/1024.0)-i32(targetImage[p-base]);let b=quantize(f32(old+delta)/1024.0)-i32(targetImage[p-base]);ds+=u32(b*b)-u32(a*a);}
 ri[tid]=ds;workgroupBarrier();for(var d=16u;d>0u;d/=2u){if(tid<d){ri[tid]+=ri[tid+d];}workgroupBarrier();}
 if(tid==0u){let newSSE=states[c].sse+ri[0];let delta=(f32(newSSE)-f32(states[c].sse))/f32(P);yes=select(0u,1u,acceptU<min(1.0,exp(-delta/temperature)));if(yes!=0u){states[c].sse=newSSE;states[c].accepted++;gs[c*cfg.N+chosen]=newg;}states[c].iteration++;}
 workgroupBarrier();if(yes!=0u){for(var q=i32(tid);q<rect.z*rect.w;q+=32){let x=rect.x+q%rect.z;let y=rect.y+q/rect.z;atomicAdd(&sums[base+u32(y)*cfg.W+u32(x)],fixed(value(newg,x,y))-fixed(value(oldg,x,y)));}}
 storageBarrier();workgroupBarrier();
}
}
@compute @workgroup_size(32)
fn adam(@builtin(local_invocation_index) tid:u32,@builtin(workgroup_id) bid:vec3u ${subgroups?',@builtin(subgroup_size) sgSize:u32':''}){
let index=bid.y*cfg.N+bid.x;let g=gs[index];let b=bounds(g);var ga=vec4f(0);var gb=vec2f(0);
for(var q=i32(tid);q<b.z*b.w;q+=32){let x=b.x+q%b.z;let y=b.y+q/b.z;let dx=f32(x)-g.x;let dy=f32(y)-g.y;let u=dx*g.co-dy*g.si;let v=dx*g.si+dy*g.co;let ux=u*g.ix;let vy=v*g.iy;let qq=ux*ux+vy*vy;
if(qq<=ROI_LIMIT){var e:f32;
${cached?'if(u32(q)<cfg.cacheStride){e=f32(expCache[index*cfg.cacheStride+u32(q)]);}else{e=exponent(qq);}':'e=exponent(qq);'}
let f=amplitude(g.a,e);let residual=f32(atomicLoad(&sums[bid.y*cfg.W*cfg.H+u32(y)*cfg.W+u32(x)]))/1024.0-targetImage[u32(y)*cfg.W+u32(x)];let d=2.0*residual*f;let ix2=g.ix*g.ix;let iy2=g.iy*g.iy;
ga+=d*vec4f(u*g.co*ix2+v*g.si*iy2,-u*g.si*ix2+v*g.co*iy2,ux*ux,vy*vy);gb+=vec2f(d*u*v*(iy2-ix2),2.0*residual*e);}}
${reduce}
if(tid==0u){var p=array<f32,6>(g.x,g.y,log(g.sx),log(g.sy),g.angle,g.a);let grad=array<f32,6>(ga.x,ga.y,ga.z,ga.w,gb.x,gb.y);let rates=array<f32,6>(0.15,0.15,0.025,0.025,0.025,1.0);
for(var k=0u;k<6u;k++){let at=index*6u+k;let gr=grad[k];let old=moments[at];let m=0.9*old.x+0.1*gr;let v=0.999*old.y+0.001*gr*gr;moments[at]=vec2f(m,v);p[k]-=rates[k]*cfg.rate*(m/cfg.c1)/(sqrt(v/cfg.c2)+1e-8);}
var out=g;out.x=clamp(p[0],0.0,f32(cfg.W));out.y=clamp(p[1],0.0,f32(cfg.H));out.sx=clamp(exp(p[2]),0.1,32.0);out.sy=clamp(exp(p[3]),0.1,32.0);out.angle=p[4];out.a=clamp(p[5],0.0,255.0);gs[index]=pack(out);
}}
`;
}

export function hybridInitial(target,W,H,N,C,roiSigma=6){
 const out=new Float32Array(N*C*12),cdf=new Float64Array(target.length);let mass=0;
 target.forEach((v,p)=>cdf[p]=(mass+=v));if(!mass)throw Error('目标不能全黑');
 for(let c=0;c<C;c++){let state=42+c*1009;const random=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296;};
 for(let j=0;j<N;j++){const q=(j+random())*mass/N;let lo=0,hi=cdf.length-1;while(lo<hi){let mid=(lo+hi)>>1;if(cdf[mid]<q)lo=mid+1;else hi=mid;}
 const sx=Math.fround(1.1+random()*1.5),sy=Math.fround(1.1+random()*1.5),a=Math.fround(random()*2*Math.PI),co=Math.cos(-a),si=Math.sin(-a);
 out.set([lo%W,Math.floor(lo/W),sx,sy,a,Math.min(230,mass/(N*2*Math.PI*sx*sy)),co,si,1/sx,1/sy,roiSigma*Math.sqrt(sx*sx*co*co+sy*sy*si*si),roiSigma*Math.sqrt(sx*sx*si*si+sy*sy*co*co)],(c*N+j)*12);
 }}return out;
}

// Deliberately full-image, no ROI truncation, no half precision, JS Float64 validation.
export function validateHybrid(data){
 const {W,H,target,gaussians}=data,P=W*H,sum=new Float64Array(P);const start=performance.now();
 for(let j=0;j<gaussians.length;j+=12){const [x,y,sx,sy,angle,a]=gaussians.slice(j,j+6),co=Math.cos(-angle),si=Math.sin(-angle);
  for(let py=0,p=0;py<H;py++)for(let px=0;px<W;px++,p++){const dx=px-x,dy=py-y,u=(dx*co-dy*si)/sx,v=(dx*si+dy*co)/sy;sum[p]+=a*Math.exp(-.5*(u*u+v*v));}}
 let sse=0,mae=0,maxError=0;const image=new Uint8Array(P);
 for(let p=0;p<P;p++){const s=sum[p];if(!Number.isFinite(s))throw Error('非有限参数或渲染值');image[p]=Math.min(255,Math.max(0,Math.floor(s+.5)));sse+=(image[p]-target[p])**2;const e=Math.abs(image[p]-data.image[p]);mae+=e;maxError=Math.max(maxError,e);}
 return {image,loss:sse/P,psnr:10*Math.log10(65025/(sse/P)),render_mae:mae/P,render_max_error:maxError,validation_ms:performance.now()-start};
}

export class HybridWebGPU {
 static async create({roiSigma=6}={}){
  if(!Number.isFinite(roiSigma)||roiSigma<3||roiSigma>6)throw Error("ROI sigma must be between 3 and 6");
  const start=performance.now();if(!navigator.gpu)throw Error('没有 WebGPU：请在支持的浏览器中通过 localhost 打开报告。');
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('没有可用 GPU adapter');
  const half=adapter.features.has('shader-f16'),subgroups=adapter.features.has('subgroups'),timestamp=adapter.features.has('timestamp-query');
  const device=await adapter.requestDevice({requiredFeatures:[...(half?['shader-f16']:[]),...(subgroups?['subgroups']:[]),...(timestamp?['timestamp-query']:[])]});
  const layout=device.createBindGroupLayout({entries:Array.from({length:7},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0?'uniform':binding===4?'read-only-storage':'storage',...(binding===0?{hasDynamicOffset:true,minBindingSize:48}:{})}}))});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});const pipelines={};
  device.pushErrorScope('validation');
  try{
   for(const mode of ['fp32',...(half?['fp16','fp16-cache']:[])]){const module=device.createShaderModule({code:hybridShader(mode!=='fp32',subgroups,mode==='fp16-cache',roiSigma)}),info=await module.getCompilationInfo();
    const errors=info.messages.filter(x=>x.type==='error');if(errors.length)throw Error(errors.map(x=>x.message+' @'+x.lineNum).join('\n'));
    pipelines[mode]=Object.fromEntries(await Promise.all(['clear','render','loss','mcmc','adam'].map(async entryPoint=>[entryPoint,await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})])));
   }
   const error=await device.popErrorScope();if(error)throw Error(error.message);
  }catch(error){device.destroy();throw error;}
  const engine=new HybridWebGPU();Object.assign(engine,{device,adapter,layout,pipelines,half,subgroups,timestamp,roiSigma,compile_ms:performance.now()-start,cancelled:false});
  device.lost.then(info=>{engine.lost=info.message||'GPU device lost';});
  return engine;
 }
 cancel(){this.cancelled=true;}
 async run(target,{W=320,H=128,N=1600,C=8,mcmc=10000,adam=8000,mode='fp32',polish=1000},onProgress=()=>{}){
  for(const [key,value,min,max] of [['N',N,1,10000],['C',C,1,16],['mcmc',mcmc,0,10000000],['adam',adam,0,10000000]])if(!Number.isInteger(value)||value<min||value>max)throw Error(`${key} 超出范围`);
  if(!mcmc&&!adam)throw Error('至少运行一步');if(W*H>65536||target.length!==W*H)throw Error('目标尺寸超限');if(!['fp32','fp16','fp16-cache'].includes(mode))throw Error('未知精度模式');if(mode!=='fp32'&&!this.half)throw Error('设备不支持 shader-f16，请选 FP32');
  if(this.lost)throw Error(this.lost);this.cancelled=false;
  const d=this.device,P=W*H,owned=[],start=performance.now();let gpuMs=0,mcmcDone=0,adamDone=0,lastPreview=0;
  const buffer=(size,usage,data)=>{const b=d.createBuffer({size:Math.ceil(size/4)*4,usage});owned.push(b);if(data)d.queue.writeBuffer(b,0,data);return b;};
  const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
  const params=hybridInitial(target,W,H,N,C,this.roiSigma),state=new Uint32Array(C*4);for(let c=0;c<C;c++)state[c*4]=101+c*1013;
  const cfg=buffer(256*257,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),gs=buffer(params.byteLength,usage,params),sums=buffer(C*P*4,usage),states=buffer(C*16,usage,state),tar=buffer(P*4,usage,Float32Array.from(target)),moments=buffer(C*N*48,usage);
  const cacheStride=mode==='fp16-cache'?Math.min(1024,Math.floor(64*1048576/(C*N*2)/32)*32):0,expCache=buffer(Math.max(4,C*N*cacheStride*2),usage);
  const bindings=[cfg,gs,sums,states,tar,moments,expCache],bind=d.createBindGroup({layout:this.layout,entries:bindings.map((b,binding)=>({binding,resource:{buffer:b,...(binding===0?{size:48}:{})}}))});
  const read=buffer(C*P*4+C*16+params.byteLength,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
  let query=null,resolve=null,tread=null;if(this.timestamp){query=d.createQuerySet({type:'timestamp',count:2});resolve=buffer(16,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC);tread=buffer(16,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);}
  const config=(slot,step,total,batch,offset,all)=>{const u=new Uint32Array(all,slot*256,12),f=new Float32Array(all,slot*256,12);u.set([W,H,N,C,step,total,batch,offset]);f[8]=.05+.95*.5*(1+Math.cos(Math.PI*step/Math.max(1,total-1)));f[9]=1-Math.pow(.9,step+1);f[10]=1-Math.pow(.999,step+1);u[11]=cacheStride;};
  const submit=async(encode,conf)=>{d.queue.writeBuffer(cfg,0,conf);const en=d.createCommandEncoder(),pass=en.beginComputePass(query?{timestampWrites:{querySet:query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});encode(pass);pass.end();if(query){en.resolveQuerySet(query,0,2,resolve,0);en.copyBufferToBuffer(resolve,0,tread,0,16);}d.queue.submit([en.finish()]);
   if(query){await tread.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(tread.getMappedRange());gpuMs+=Number(t[1]-t[0])/1e6;tread.unmap();}else await d.queue.onSubmittedWorkDone();if(this.lost)throw Error(this.lost);
  };
  const dispatch=(pass,name,slot,precision=mode)=>{pass.setBindGroup(0,bind,[slot*256]);pass.setPipeline(this.pipelines[precision][name]);if(name==='clear')pass.dispatchWorkgroups(Math.ceil(C*P/256));else if(name==='render'||name==='adam')pass.dispatchWorkgroups(N,C);else pass.dispatchWorkgroups(C);};
  const snapshot=async(final=false)=>{const en=d.createCommandEncoder();en.copyBufferToBuffer(sums,0,read,0,C*P*4);en.copyBufferToBuffer(states,0,read,C*P*4,C*16);if(final)en.copyBufferToBuffer(gs,0,read,C*P*4+C*16,params.byteLength);d.queue.submit([en.finish()]);await read.mapAsync(GPUMapMode.READ);const a=read.getMappedRange().slice(0);read.unmap();const sum=new Uint32Array(a,0,C*P),st=new Uint32Array(a,C*P*4,C*4);let winner=0;
   for(let c=1;c<C;c++)if(st[c*4+2]<st[winner*4+2])winner=c;
   const image=Uint8Array.from(sum.subarray(winner*P,(winner+1)*P),x=>Math.min(255,Math.floor(x/1024+.5)));
   return {W,H,N,C,mcmc,adam,mode,polish,roi_sigma:this.roiSigma,fp16_steps:mode==='fp32'?0:Math.min(adamDone,Math.max(0,adam-polish)),cache_stride:cacheStride,mcmc_done:mcmcDone,adam_done:adamDone,winner,image,cached_loss:st[winner*4+2]/P,accepted:st[winner*4+1],gpu_ms:query?gpuMs:null,fit_wall_ms:performance.now()-start,workspace_bytes:owned.reduce((s,b)=>s+b.size,0),...(final?{gaussians:Array.from(new Float32Array(a,C*P*4+C*16+winner*N*48,N*12))}:{}),adapter:{vendor:this.adapter.info?.vendor,architecture:this.adapter.info?.architecture},features:{half:this.half,subgroups:this.subgroups,timestamp:this.timestamp},cancelled:this.cancelled};
  };
  d.pushErrorScope('validation');
  try{
   let conf=new ArrayBuffer(256);config(0,0,Math.max(1,mcmc),0,0,conf);
   await submit(pass=>{dispatch(pass,'clear',0,'fp32');dispatch(pass,'render',0,'fp32');dispatch(pass,'loss',0,'fp32');},conf);
   while(mcmcDone<mcmc&&!this.cancelled){const batch=Math.min(256,mcmc-mcmcDone);conf=new ArrayBuffer(256);config(0,0,mcmc,batch,mcmcDone,conf);await submit(pass=>dispatch(pass,'mcmc',0,'fp32'),conf);mcmcDone+=batch;
    if(performance.now()-lastPreview>250){await onProgress(await snapshot());lastPreview=performance.now();}
   }
   while(adamDone<adam&&!this.cancelled){const batch=Math.min(128,adam-adamDone);conf=new ArrayBuffer(batch*256);for(let i=0;i<batch;i++)config(i,adamDone+i,adam,0,0,conf);
    await submit(pass=>{for(let i=0;i<batch;i++){const precision=adamDone+i>=adam-polish?'fp32':mode;dispatch(pass,'clear',i,precision);dispatch(pass,'render',i,precision);dispatch(pass,'adam',i,precision);}},conf);adamDone+=batch;
    if(performance.now()-lastPreview>250){const c=new ArrayBuffer(256);config(0,0,1,0,0,c);await submit(pass=>{dispatch(pass,'clear',0,'fp32');dispatch(pass,'render',0,'fp32');dispatch(pass,'loss',0,'fp32');},c);await onProgress(await snapshot());lastPreview=performance.now();}
   }
   conf=new ArrayBuffer(256);config(0,0,1,0,0,conf);await submit(pass=>{dispatch(pass,'clear',0,'fp32');dispatch(pass,'render',0,'fp32');dispatch(pass,'loss',0,'fp32');},conf);
   const result=await snapshot(true);const error=await d.popErrorScope();if(error)throw Error(error.message);return result;
  }finally{query?.destroy();for(const b of owned)b.destroy();}
 }
 destroy(){this.device.destroy();}
}
