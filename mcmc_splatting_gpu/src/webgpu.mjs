// WebGPU persistent-chain prototype. FP16 optional; parameters/sums stay FP32 and SSE uses u32.
export function makeShader(mode) {
 const half=mode!=='fp32', C=half?'f16':'f32';
 return (half?'enable f16;\n':'')+`
struct G { x:f32,y:f32,sx:f32,sy:f32,angle:f32,a:f32,co:f32,si:f32 };
struct State { rng:u32, accepted:u32, sse:u32, iteration:u32 };
struct Config { W:u32,H:u32,N:u32,C:u32,batch:u32,total:u32,pad0:u32,pad1:u32,T0:f32,T1:f32,shrink:u32,pad2:u32 };
@group(0) @binding(0) var<uniform> cfg:Config;
@group(0) @binding(1) var<storage,read_write> gs:array<G>;
@group(0) @binding(2) var<storage,read_write> sums:array<f32>;
@group(0) @binding(3) var<storage,read_write> cache:array<__CACHE__>;
@group(0) @binding(4) var<storage,read_write> candidates:array<__CACHE__>;
@group(0) @binding(5) var<storage,read_write> states:array<State>;
@group(0) @binding(6) var<storage,read> target:array<u32>;
var<workgroup> g:G;
var<workgroup> changed:u32;
var<workgroup> decision:u32;
var<workgroup> acceptU:f32;
var<workgroup> temperature:f32;
var<workgroup> reduce:array<u32,256>;
fn next(s:u32)->u32 {var v=s;v=v^(v<<13u);v=v^(v>>17u);v=v^(v<<5u);return v;}
fn uniform(s:u32)->f32 {return f32(s>>8u)*(1.0/16777216.0);}
fn quantize(s:f32)->i32{return i32(clamp(floor(s+0.5),0.0,255.0));}
fn value(p:u32,a:G)->f32 {
 let dx=f32(p%cfg.W)-a.x;let dy=f32(p/cfg.W)-a.y;
 let u=(dx*a.co-dy*a.si)/a.sx;let v=(dx*a.si+dy*a.co)/a.sy;
 let exponent=-0.5*(u*u+v*v);
 return __VALUE__;
}
@compute @workgroup_size(256)
fn init_pixels(@builtin(global_invocation_id) id:vec3<u32>){
 let p=id.x;let chain=id.y;let P=cfg.W*cfg.H;if(p>=P||chain>=cfg.C){return;}
 var total=0.0;for(var j=0u;j<cfg.N;j++){
   let v=__CACHE__(value(p,gs[chain*cfg.N+j]));
   cache[(chain*cfg.N+j)*P+p]=v;total+=f32(v);
 }sums[chain*P+p]=total;
}
@compute @workgroup_size(256)
fn init_loss(@builtin(local_invocation_index) tid:u32,@builtin(workgroup_id) bid:vec3<u32>){
 let chain=bid.x;let P=cfg.W*cfg.H;var s=0u;
 for(var p=tid;p<P;p+=256u){let d=quantize(sums[chain*P+p])-i32(target[p]);s+=u32(d*d);}
 reduce[tid]=s;workgroupBarrier();
 for(var stride=128u;stride>0u;stride/=2u){if(tid<stride){reduce[tid]+=reduce[tid+stride];}workgroupBarrier();}
 if(tid==0u){states[chain].sse=reduce[0];}
}
@compute @workgroup_size(256)
fn iterate(@builtin(local_invocation_index) tid:u32,@builtin(workgroup_id) bid:vec3<u32>){
 let chain=bid.x;let P=cfg.W*cfg.H;let base=chain*P;
 for(var it=0u;it<cfg.batch;it++){
   if(tid==0u){
     var s=next(states[chain].rng);changed=min(u32(uniform(s)*f32(cfg.N)),cfg.N-1u);
     s=next(s);let param=min(u32(uniform(s)*6.0),5u);s=next(s);let delta=(uniform(s)-0.5)*2.0;s=next(s);acceptU=uniform(s);states[chain].rng=s;
     let ratio=f32(states[chain].iteration)/f32(max(cfg.total-1u,1u));
     temperature=cfg.T0*pow(cfg.T1/cfg.T0,ratio);
     var scale=1.0;if(cfg.shrink!=0u){scale=0.15+0.85*pow(cfg.T1/cfg.T0,ratio*0.25);}
     g=gs[chain*cfg.N+changed];
     switch(param){
       case 0u:{g.x=clamp(g.x+delta*2.0*scale,0.0,f32(cfg.W));}
       case 1u:{g.y=clamp(g.y+delta*2.0*scale,0.0,f32(cfg.H));}
       case 2u:{g.sx=max(0.1,g.sx+delta*scale);}
       case 3u:{g.sy=max(0.1,g.sy+delta*scale);}
       case 4u:{g.angle+=delta*0.1*scale;}
       default:{g.a=clamp(g.a+delta*10.0*scale,0.0,255.0);}
     }g.co=cos(-g.angle);g.si=sin(-g.angle);
   }
   workgroupBarrier();var sse=0u;let old=(chain*cfg.N+changed)*P;
   for(var p=tid;p<P;p+=256u){
     let fresh=__CACHE__(value(p,g));candidates[base+p]=fresh;
     let total=sums[base+p]-f32(cache[old+p])+f32(fresh);
     let d=quantize(total)-i32(target[p]);sse+=u32(d*d);
   }
   reduce[tid]=sse;workgroupBarrier();
   for(var stride=128u;stride>0u;stride/=2u){if(tid<stride){reduce[tid]+=reduce[tid+stride];}workgroupBarrier();}
   if(tid==0u){
     let delta=(f32(reduce[0])-f32(states[chain].sse))/f32(P);
     decision=select(0u,1u,acceptU<min(1.0,exp(-delta/temperature)));
     if(decision!=0u){states[chain].sse=reduce[0];states[chain].accepted++;gs[chain*cfg.N+changed]=g;}
     states[chain].iteration++;
   }
   workgroupBarrier();storageBarrier();
   if(decision!=0u){for(var p=tid;p<P;p+=256u){let v=candidates[base+p];sums[base+p]=sums[base+p]-f32(cache[old+p])+f32(v);cache[old+p]=v;}}
   storageBarrier();workgroupBarrier();
 }
}`.replaceAll('__CACHE__',C).replaceAll('__VALUE__',mode==='f16-exp'?'f32(f16(a.a)*exp(f16(max(exponent,-80.0))))':'a.a*exp(exponent)');
}
export function seededGaussians(target,W,H,N,C=1){
 const out=new Float32Array(N*C*8),cdf=new Float64Array(target.length);let mass=0;
 target.forEach((v,p)=>cdf[p]=(mass+=v));if(!mass)throw Error('目标不能全黑');
 for(let c=0;c<C;c++){const r=rng(42+c*1009);for(let j=0;j<N;j++){
   const q=(j+r())*mass/N;let lo=0,hi=cdf.length-1;while(lo<hi){const mid=(lo+hi)>>1;if(cdf[mid]<q)lo=mid+1;else hi=mid;}
   const sx=1.1+r()*1.5,sy=1.1+r()*1.5,angle=r()*2*Math.PI,a=Math.min(230,mass/(N*2*Math.PI*sx*sy));
   out.set([lo%W,Math.floor(lo/W),sx,sy,angle,a,Math.cos(-angle),Math.sin(-angle)],(c*N+j)*8);
 }}return out;
}
export class WebGPUFit {
 static async create(mode='fp32'){
   if(!navigator.gpu)throw Error('此浏览器没有 WebGPU；请使用支持 WebGPU 的浏览器，并在 localhost 打开');
   const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('没有可用 GPU adapter');
   if(mode!=='fp32'&&!adapter.features.has('shader-f16'))throw Error('不支持 shader-f16，请用 FP32');
   const timestamp=adapter.features.has('timestamp-query'),features=[...(mode==='fp32'?[]:['shader-f16']),...(timestamp?['timestamp-query']:[])];
   const device=await adapter.requestDevice({requiredFeatures:features,requiredLimits:{maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:adapter.limits.maxBufferSize}});
   device.pushErrorScope('validation');const module=device.createShaderModule({code:makeShader(mode)});const info=await module.getCompilationInfo();
   if(info.messages.some(m=>m.type==='error')){device.destroy();throw Error(info.messages.filter(m=>m.type==='error').map(m=>m.message+' @'+m.lineNum).join('\n'));}
   const layout=device.createBindGroupLayout({entries:Array.from({length:7},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0?'uniform':binding===6?'read-only-storage':'storage'}}))});
   const pl=device.createPipelineLayout({bindGroupLayouts:[layout]});const pipelines={};
   for(const entryPoint of ['init_pixels','init_loss','iterate'])pipelines[entryPoint]=await device.createComputePipelineAsync({layout:pl,compute:{module,entryPoint}});
   const error=await device.popErrorScope();if(error){device.destroy();throw Error(error.message);}
   return new WebGPUFit(device,adapter,mode,timestamp,layout,pipelines);
 }
 constructor(device,adapter,mode,timestamp,layout,pipelines){Object.assign(this,{device,adapter,mode,timestamp,layout,pipelines});this.buffers=[];this.cancelled=false;}
 buffer(size,usage,data){if(size>this.adapter.limits.maxBufferSize)throw Error('GPU buffer 超限');const b=this.device.createBuffer({size:Math.ceil(size/4)*4,usage});this.buffers.push(b);if(data)this.device.queue.writeBuffer(b,0,data);return b;}
 async run(target,W,H,N,C,total,onProgress=()=>{}){
   if(W*H>65536)throw Error('此原型整数 SSE 限制最多 65,536 像素');if(total%64)throw Error('迭代数须为64的倍数');
   const d=this.device,P=W*H,half=this.mode!=='fp32',bytes=C*N*P*(half?2:4);
   if(bytes>this.adapter.limits.maxStorageBufferBindingSize)throw Error('贡献矩阵超过浏览器单 buffer 限制，请减少高斯 / 链 / 分辨率');
   this.cancelled=false;const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
   const params=seededGaussians(target,W,H,N,C),state=new Uint32Array(C*4);for(let c=0;c<C;c++)state[c*4]=101+c*1013;
   const conf=new ArrayBuffer(48),ui=new Uint32Array(conf),ff=new Float32Array(conf);ui.set([W,H,N,C,64,total,0,0]);ff[8]=2;ff[9]=.001;ui[10]=1;
   const buffers=[this.buffer(48,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,conf),this.buffer(params.byteLength,storage,params),this.buffer(C*P*4,storage),this.buffer(bytes,storage),this.buffer(C*P*(half?2:4),storage),this.buffer(state.byteLength,storage,state),this.buffer(P*4,storage,Uint32Array.from(target))];
   const bind=d.createBindGroup({layout:this.layout,entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
   const read=this.buffer(C*P*4+state.byteLength+params.byteLength,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
   let queries=null,resolve=null,tread=null;if(this.timestamp){queries=d.createQuerySet({type:'timestamp',count:2});resolve=this.buffer(16,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC);tread=this.buffer(16,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);}
   const beginSetup=performance.now();let encoder=d.createCommandEncoder(),pass=encoder.beginComputePass();pass.setBindGroup(0,bind);pass.setPipeline(this.pipelines.init_pixels);pass.dispatchWorkgroups(Math.ceil(P/256),C);pass.setPipeline(this.pipelines.init_loss);pass.dispatchWorkgroups(C);pass.end();d.queue.submit([encoder.finish()]);await d.queue.onSubmittedWorkDone();const setupMs=performance.now()-beginSetup;
   const start=performance.now();let gpuMs=0,iteration=0,last=null;
   const snapshot=async()=>{
     const en=d.createCommandEncoder();en.copyBufferToBuffer(buffers[2],0,read,0,C*P*4);en.copyBufferToBuffer(buffers[5],0,read,C*P*4,state.byteLength);en.copyBufferToBuffer(buffers[1],0,read,C*P*4+state.byteLength,params.byteLength);d.queue.submit([en.finish()]);
     await read.mapAsync(GPUMapMode.READ);const copy=read.getMappedRange().slice(0);read.unmap();
     const sums=new Float32Array(copy,0,C*P),stats=new Uint32Array(copy,C*P*4,C*4),gs=new Float32Array(copy,C*P*4+state.byteLength);
     let winner=0;for(let c=1;c<C;c++)if(stats[c*4+2]<stats[winner*4+2])winner=c;
     return {W,H,N,C,total,mode:this.mode,iteration:stats[winner*4+3],accepted:stats[winner*4+1],loss:stats[winner*4+2]/P,image:Array.from(sums.subarray(winner*P,(winner+1)*P),s=>Math.min(255,Math.max(0,Math.floor(s+.5)))),gaussians:Array.from(gs.subarray(winner*N*8,(winner+1)*N*8)),winner,gpu_ms:this.timestamp?gpuMs:null,wall_ms:performance.now()-start,setup_ms:setupMs,cache_bytes:bytes,features:Array.from(this.adapter.features),adapter:this.adapter.info?{vendor:this.adapter.info.vendor,architecture:this.adapter.info.architecture,device:this.adapter.info.device,description:this.adapter.info.description}:null};
   };
   try{
     while(iteration<total&&!this.cancelled){
       encoder=d.createCommandEncoder();pass=encoder.beginComputePass(this.timestamp?{timestampWrites:{querySet:queries,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});
       pass.setBindGroup(0,bind);
       if(iteration>0&&iteration%8192===0){pass.setPipeline(this.pipelines.init_pixels);pass.dispatchWorkgroups(Math.ceil(P/256),C);pass.setPipeline(this.pipelines.init_loss);pass.dispatchWorkgroups(C);}
       pass.setPipeline(this.pipelines.iterate);const count=Math.min(1024,total-iteration)/64;for(let i=0;i<count;i++)pass.dispatchWorkgroups(C);pass.end();
       if(this.timestamp){encoder.resolveQuerySet(queries,0,2,resolve,0);encoder.copyBufferToBuffer(resolve,0,tread,0,16);}d.queue.submit([encoder.finish()]);
       if(this.timestamp){await tread.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(tread.getMappedRange());gpuMs+=Number(t[1]-t[0])/1e6;tread.unmap();}else await d.queue.onSubmittedWorkDone();
       iteration+=count*64;last=await snapshot();await onProgress(last);
     }
     return last;
   }finally{if(queries)queries.destroy();for(const b of this.buffers)b.destroy();this.buffers=[];}
 }
 destroy(){this.cancelled=true;this.device.destroy();}
}

