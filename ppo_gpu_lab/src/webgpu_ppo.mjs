/* Real FP32 WebGPU PPO. No ML runtime and no CPU gradient fallback.
 * Flat parameter order matches ppo_v2.py (418 actor + 401 critic).
 * CPU only encodes commands and Fisher-Yates permutations; model, rollouts,
 * GAE, activations, gradients and Adam moments stay in GPU buffers.
 */
export function ppoShader({n=256,horizon=64,minibatch=2048}={}){
return `
const N:u32=${n}u; const T:u32=${horizon}u; const B:u32=N*T; const MB:u32=${minibatch}u;
struct Env {s:array<f32,6>,age:u32,rng:u32};
struct Config {force:f32,angle:f32,kick:f32,robust:f32,actorLR:f32,criticLR:f32,entropy:f32,pad:f32};
struct Update {offset:u32,step:u32,pad0:u32,pad1:u32};
@group(0) @binding(0) var<storage,read_write> w:array<f32>;
@group(0) @binding(1) var<storage,read_write> env:array<Env>;
@group(0) @binding(2) var<storage,read_write> roll:array<f32>;
@group(0) @binding(3) var<storage,read_write> act:array<f32>;
@group(0) @binding(4) var<storage,read_write> grad:array<f32>;
@group(0) @binding(5) var<storage,read_write> opt:array<f32>;
@group(0) @binding(6) var<storage,read> indices:array<u32>;
@group(0) @binding(7) var<uniform> cfg:Config;
@group(0) @binding(8) var<uniform> upd:Update;
var<workgroup> scratch:array<f32,256>;
fn random(r:ptr<function,u32>)->f32 {var x=*r;x=x^(x<<13u);x=x^(x>>17u);x=x^(x<<5u);*r=x;return (f32(x>>8u)+0.5)*(1.0/16777216.0);}
fn reset(r:ptr<function,u32>)->array<f32,6>{
  var s:array<f32,6>;let wide=cfg.robust>0.5&&random(r)<0.6;
  let a=select(0.01,cfg.angle,wide);let x=select(0.05,0.25,wide);
  s[0]=(random(r)*2.0-1.0)*x;s[1]=(random(r)*2.0-1.0)*select(0.05,0.15,wide);
  s[2]=(random(r)*2.0-1.0)*a;s[3]=(random(r)*2.0-1.0)*select(0.025,0.1,wide);
  s[4]=(random(r)*2.0-1.0)*a;s[5]=(random(r)*2.0-1.0)*select(0.025,0.1,wide);return s;
}
fn network(s:array<f32,6>,critic:bool)->vec3<f32>{
  let base=select(0u,418u,critic);let outputs=select(2u,1u,critic);
  var h1:array<f32,16>;var h2:array<f32,16>;
  for(var j=0u;j<16u;j++){var z=w[base+96u+j];for(var k=0u;k<6u;k++){z+=w[base+j*6u+k]*s[k];}h1[j]=max(z,0.0);}
  for(var j=0u;j<16u;j++){var z=w[base+368u+j];for(var k=0u;k<16u;k++){z+=w[base+112u+j*16u+k]*h1[k];}h2[j]=max(z,0.0);}
  var out=vec3<f32>(0.0);for(var j=0u;j<outputs;j++){var z=w[base+384u+outputs*16u+j];for(var k=0u;k<16u;k++){z+=w[base+384u+j*16u+k]*h2[k];}out[j]=z;}return out;
}
fn physical(s:array<f32,6>,force:f32)->array<f32,6>{
  let x=s[0];let v=s[1];let t1=s[2];let w1=s[3];let t2=s[4];let w2=s[5];
  let b=0.15*cos(t1);let c=0.05*cos(t2);let e=0.05*cos(t1-t2);
  let n0=force+0.15*sin(t1)*w1*w1+0.05*sin(t2)*w2*w2;
  let n1=1.4715*sin(t1)-0.05*sin(t1-t2)*w2*w2;
  let n2=0.4905*sin(t2)+0.05*sin(t1-t2)*w1*w1;
  let l10=b/1.2;let l20=c/1.2;let d1=0.1275-b*l10;let l21=(e-c*l10)/d1;let d2=0.0275-c*l20-l21*l21*d1;
  let y1=n1-l10*n0;let y2=n2-l20*n0-l21*y1;let z2=y2/d2;let z1=y1/d1-l21*z2;let z0=n0/1.2-l10*z1-l20*z2;
  let nv=v+0.02*z0;let nw1=w1+0.02*z1;let nw2=w2+0.02*z2;
  var out:array<f32,6>;out[0]=x+0.02*nv;out[1]=nv;out[2]=t1+0.02*nw1;out[3]=nw1;out[4]=t2+0.02*nw2;out[5]=nw2;
  // Remainder convention matches ppo.py. Terminal angles are much smaller than pi.
  for(var j=2u;j<=4u;j+=2u){let a=out[j]+3.141592653589793;out[j]=a-trunc(a/6.283185307179586)*6.283185307179586-3.141592653589793;}return out;
}
fn logprob(raw:f32,mu:f32,ls:f32)->f32 {let z=(raw-mu)*exp(-ls);return -0.5*z*z-ls-0.9189385332046727;}
@compute @workgroup_size(32)
fn rollout(@builtin(global_invocation_id) id:vec3<u32>){
  let e=id.x;if(e>=N){return;}var s=env[e].s;var age=env[e].age;var rng=env[e].rng;
  for(var t=0u;t<T;t++){
    let out=network(s,false);let mu=out.x;let ls=clamp(out.y,-5.0,2.0);let value=network(s,true).x;
    let noise=sqrt(-2.0*log(random(&rng)))*cos(6.283185307179586*random(&rng));
    let raw=mu+exp(ls)*noise;let force=clamp(raw,-1.0,1.0)*cfg.force;
    var ns=physical(s,force);age++;
    // The impulse belongs to this transition: bootstrap and the next observation
    // must use the exact same perturbed state, including across rollout blocks.
    if(cfg.robust>0.5&&age>20u&&random(&rng)<0.003){ns[1]+=(random(&rng)*2.0-1.0)*cfg.kick;}
    let threshold=select(0.20943951023931953,1.0471975511965976,cfg.robust>0.5);
    let terminal=abs(ns[0])>2.4||abs(ns[2])>threshold||abs(ns[4])>threshold;
    let ended=terminal||age>=1000u;
    let angleCost=select(0.5,4.0,cfg.robust>0.5);
    var reward=1.0-angleCost*(ns[2]*ns[2]+ns[4]*ns[4])-0.1*ns[0]*ns[0]-0.01*(ns[1]*ns[1]+ns[3]*ns[3]+ns[5]*ns[5]);
    if(terminal){reward=-100.0;}
    let o=(t*N+e)*16u;for(var j=0u;j<6u;j++){roll[o+j]=s[j];}
    roll[o+6u]=raw;roll[o+7u]=logprob(raw,mu,ls);roll[o+8u]=value;roll[o+9u]=reward;
    roll[o+10u]=select(0.0,1.0,terminal);roll[o+11u]=select(0.0,1.0,ended);
    roll[o+12u]=network(ns,true).x;
    s=ns;if(ended){s=reset(&rng);age=0u;}
  }env[e].s=s;env[e].age=age;env[e].rng=rng;
}
@compute @workgroup_size(64)
fn gae(@builtin(global_invocation_id) id:vec3<u32>){
 let e=id.x;if(e>=N){return;}var last=0.0;
 for(var tt=i32(T)-1;tt>=0;tt--){let o=(u32(tt)*N+e)*16u;let delta=roll[o+9u]+0.99*roll[o+12u]*(1.0-roll[o+10u])-roll[o+8u];last=delta+0.99*0.95*(1.0-roll[o+11u])*last;roll[o+13u]=last;roll[o+14u]=last+roll[o+8u];}
}
@compute @workgroup_size(64)
fn stats_partial(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) gid:vec3<u32>){
 var sum=0.0;var sq=0.0;var reward=0.0;var done=0.0;
 for(var j=lane;j<256u;j+=64u){let i=gid.x*256u+j;if(i<B){let a=roll[i*16u+13u];sum+=a;sq+=a*a;reward+=roll[i*16u+9u];done+=roll[i*16u+10u];}}
 scratch[lane]=sum;scratch[64u+lane]=sq;scratch[128u+lane]=reward;scratch[192u+lane]=done;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){for(var k=0u;k<4u;k++){scratch[k*64u+lane]+=scratch[k*64u+lane+stride];}}workgroupBarrier();}
 if(lane==0u){for(var k=0u;k<4u;k++){grad[gid.x*4u+k]=scratch[k*64u];}}
}
@compute @workgroup_size(64)
fn stats_final(@builtin(local_invocation_index) lane:u32){
 for(var k=0u;k<4u;k++){var sum=0.0;for(var j=lane;j<(B+255u)/256u;j+=64u){sum+=grad[j*4u+k];}scratch[k*64u+lane]=sum;}workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){for(var k=0u;k<4u;k++){scratch[k*64u+lane]+=scratch[k*64u+lane+stride];}}workgroupBarrier();}
 if(lane==0u){let mean=scratch[0]/f32(B);roll[B*16u]=mean;roll[B*16u+1u]=sqrt(max(0.0,scratch[64u]/f32(B)-mean*mean))+1e-8;roll[B*16u+2u]=scratch[128u]/f32(B);roll[B*16u+3u]=scratch[192u];}
}
@compute @workgroup_size(256)
fn normalize(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i<B){roll[i*16u+13u]=(roll[i*16u+13u]-roll[B*16u])/roll[B*16u+1u];}}
@compute @workgroup_size(64)
fn forward_backward(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=MB){return;}let o=indices[upd.offset+i]*16u;var s:array<f32,6>;for(var j=0u;j<6u;j++){s[j]=roll[o+j];act[j*MB+i]=s[j];}
 for(var net=0u;net<2u;net++){
  let base=select(0u,418u,net==1u);let outputs=select(2u,1u,net==1u);
  var h1:array<f32,16>;var h2:array<f32,16>;var out:array<f32,2>;var dout:array<f32,2>;var dh1:array<f32,16>;var dh2:array<f32,16>;
  for(var j=0u;j<16u;j++){var z=w[base+96u+j];for(var k=0u;k<6u;k++){z+=w[base+j*6u+k]*s[k];}h1[j]=max(z,0.0);}
  for(var j=0u;j<16u;j++){var z=w[base+368u+j];for(var k=0u;k<16u;k++){z+=w[base+112u+j*16u+k]*h1[k];}h2[j]=max(z,0.0);}
  for(var j=0u;j<outputs;j++){var z=w[base+384u+outputs*16u+j];for(var k=0u;k<16u;k++){z+=w[base+384u+j*16u+k]*h2[k];}out[j]=z;}
  if(net==0u){
   let ls=clamp(out[1],-5.0,2.0);let raw=roll[o+6u];let lp=logprob(raw,out[0],ls);let diff=lp-roll[o+7u];let ratio=exp(clamp(diff,-50.0,50.0));let ad=roll[o+13u];
   var dl=-ad*ratio;if((ad>=0.0&&ratio>1.2)||(ad<0.0&&ratio<0.8)||abs(diff)>50.0){dl=0.0;}
   let delta=raw-out[0];let invVar=exp(-2.0*ls);dout[0]=dl*delta*invVar;
   dout[1]=dl*(delta*delta*invVar-1.0)-cfg.entropy;if(out[1]<-5.0||out[1]>2.0){dout[1]=0.0;}
   act[70u*MB+i]=dout[0];act[71u*MB+i]=dout[1];
  }else{dout[0]=out[0]-roll[o+14u];act[136u*MB+i]=dout[0];}
  for(var j=0u;j<16u;j++){var z=0.0;for(var k=0u;k<outputs;k++){z+=w[base+384u+k*16u+j]*dout[k];}dh2[j]=select(0.0,z,h2[j]>0.0);}
  for(var j=0u;j<16u;j++){var z=0.0;for(var k=0u;k<16u;k++){z+=w[base+112u+k*16u+j]*dh2[k];}dh1[j]=select(0.0,z,h1[j]>0.0);}
  let a=select(6u,72u,net==1u);for(var j=0u;j<16u;j++){act[(a+j)*MB+i]=h1[j];act[(a+16u+j)*MB+i]=h2[j];act[(a+32u+j)*MB+i]=dh1[j];act[(a+48u+j)*MB+i]=dh2[j];}
 }
}
@compute @workgroup_size(64)
fn gradients(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) gid:vec3<u32>){
 let p=gid.x;var left=0u;var right=0u;var bias=false;
 if(p<96u){left=38u+p/6u;right=p%6u;}
 else if(p<112u){left=38u+p-96u;bias=true;}
 else if(p<368u){let q=p-112u;left=54u+q/16u;right=6u+q%16u;}
 else if(p<384u){left=54u+p-368u;bias=true;}
 else if(p<416u){let q=p-384u;left=70u+q/16u;right=22u+q%16u;}
 else if(p<418u){left=70u+p-416u;bias=true;}
 else if(p<514u){let q=p-418u;left=104u+q/6u;right=q%6u;}
 else if(p<530u){left=104u+p-514u;bias=true;}
 else if(p<786u){let q=p-530u;left=120u+q/16u;right=72u+q%16u;}
 else if(p<802u){left=120u+p-786u;bias=true;}
 else if(p<818u){left=136u;right=88u+p-802u;}
 else{left=136u;bias=true;}
 var sum=0.0;for(var i=lane;i<MB;i+=64u){var v=act[left*MB+i];if(!bias){v*=act[right*MB+i];}sum+=v;}scratch[lane]=sum;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){scratch[lane]+=scratch[lane+stride];}workgroupBarrier();}
 if(lane==0u){grad[p]=scratch[0]/f32(MB);}
}
@compute @workgroup_size(64)
fn adam(@builtin(global_invocation_id) id:vec3<u32>){
 let p=id.x;if(p>=819u){return;}let g=grad[p];let m=0.9*opt[p]+0.1*g;let v=0.999*opt[819u+p]+0.001*g*g;
 opt[p]=m;opt[819u+p]=v;let bc1=1.0-pow(0.9,f32(upd.step));let bc2=1.0-pow(0.999,f32(upd.step));
 let lr=select(cfg.actorLR,cfg.criticLR,p>=418u);w[p]-=lr*(m/bc1)/(sqrt(v/bc2)+1e-8);
}
@compute @workgroup_size(64)
fn test_physics(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=MB){return;}var s:array<f32,6>;for(var j=0u;j<6u;j++){s[j]=roll[i*16u+j];}let ns=physical(s,clamp(roll[i*16u+6u],-1.0,1.0)*cfg.force);for(var j=0u;j<6u;j++){act[j*MB+i]=ns[j];}}
`;
}

export function rng32(seed){let x=seed>>>0||1;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return((x>>>8)+.5)/16777216;};}
export function exportActor(flat){const p={};let o=0;for(const [layer,input,output]of[[0,6,16],[2,16,16],[4,16,2]]){p[layer+'.weight']=Array.from({length:output},()=>{const row=Array.from(flat.slice(o,o+input));o+=input;return row;});p[layer+'.bias']=Array.from(flat.slice(o,o+output));o+=output;}return p;}

export class WebGPUPPO {
 static async create(initial,{n=256,horizon=64,minibatch=2048,epochs=10,seed=71,force=40,robust=true}={}){
  if(!navigator.gpu)throw Error('WebGPU 不可用。请使用支持 WebGPU 的浏览器，通过 localhost / HTTPS 打开。');
  const start=performance.now(),adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('没有可用 GPU adapter');
  const timestamp=adapter.features.has('timestamp-query');const device=await adapter.requestDevice({requiredFeatures:timestamp?['timestamp-query']:[]});
  device.pushErrorScope('validation');const module=device.createShaderModule({code:ppoShader({n,horizon,minibatch})});const info=await module.getCompilationInfo();
  const errors=info.messages.filter(x=>x.type==='error');if(errors.length){device.destroy();throw Error(errors.map(x=>x.message+' @ '+x.lineNum).join('\n'));}
  const entries=Array.from({length:9},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:binding<7?{type:binding===6?'read-only-storage':'storage'}:{type:'uniform',hasDynamicOffset:binding===8,minBindingSize:binding===8?16:32}}));
  const layout=device.createBindGroupLayout({entries}),pl=device.createPipelineLayout({bindGroupLayouts:[layout]}),pipelines={};
  for(const entryPoint of ['rollout','gae','stats_partial','stats_final','normalize','forward_backward','gradients','adam','test_physics'])pipelines[entryPoint]=await device.createComputePipelineAsync({layout:pl,compute:{module,entryPoint}});
  const error=await device.popErrorScope();if(error){device.destroy();throw Error(error.message);}
  const instance=new WebGPUPPO(device,adapter,layout,pipelines,{n,horizon,minibatch,epochs,seed,force,robust,timestamp});instance.reset(initial,seed);await device.queue.onSubmittedWorkDone();instance.setupMs=performance.now()-start;return instance;
 }
 constructor(device,adapter,layout,pipelines,config){
  Object.assign(this,{device,adapter,layout,pipelines,config});this.buffers=[];this.fatal=null;
  device.addEventListener('uncapturederror',e=>this.fatal=e.error.message);device.lost.then(x=>{if(x.reason!=='destroyed')this.fatal='GPU device lost: '+x.message;});
  const {n,horizon,minibatch,epochs}=config;this.b=n*horizon;this.updates=this.b/minibatch*epochs;
  if(!Number.isInteger(this.updates)||this.b%256)throw Error('Invalid batch shape');
  const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
  this.w=this.buffer(819*4,storage);this.env=this.buffer(n*32,storage);this.roll=this.buffer((this.b*16+16)*4,storage);
  this.act=this.buffer(minibatch*137*4,storage);this.grad=this.buffer(Math.max(819,this.b/256*4)*4,storage);this.opt=this.buffer(1638*4,storage);this.indices=this.buffer(this.b*epochs*4,storage);
  this.cfg=this.buffer(32,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);this.upd=this.buffer(this.updates*256,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
  this.bind=device.createBindGroup({layout,entries:[this.w,this.env,this.roll,this.act,this.grad,this.opt,this.indices,this.cfg,this.upd].map((buffer,binding)=>({binding,resource:{buffer,...(binding===8?{size:16}:{})}}))});
  this.readBuffer=this.buffer(Math.max(this.roll.size,this.act.size,819*4),GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
  if(config.timestamp){this.query=device.createQuerySet({type:'timestamp',count:2});this.queryResult=this.buffer(16,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC);this.queryRead=this.buffer(16,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);}
  this.allocatedBytes=this.buffers.reduce((a,b)=>a+b.size,0);
 }
 buffer(size,usage){const b=this.device.createBuffer({size:Math.ceil(size/4)*4,usage});this.buffers.push(b);return b;}
 reset(initial,seed=this.config.seed){
  this.iteration=0;this.rng=rng32(seed);this.device.queue.writeBuffer(this.w,0,new Float32Array(initial));this.device.queue.writeBuffer(this.opt,0,new Float32Array(1638));
  const a=new ArrayBuffer(this.config.n*32),f=new Float32Array(a),u=new Uint32Array(a);
  const scales=[.1,.1,.02,.05,.02,.05];for(let e=0;e<this.config.n;e++){for(let j=0;j<6;j++)f[e*8+j]=(this.rng()-.5)*scales[j];u[e*8+6]=0;u[e*8+7]=(seed+Math.imul(e+1,2654435761))>>>0||1;}
  this.device.queue.writeBuffer(this.env,0,a);this.writeConfig(0);
 }
 writeConfig(progress=1){const c=this.config;this.device.queue.writeBuffer(this.cfg,0,new Float32Array([c.force,(2+8*Math.min(1,progress))*Math.PI/180,.35*Math.min(1,progress),+c.robust,c.actorLR??3e-4,c.criticLR??1e-3,c.entropy??.01,0]));}
 dispatch(pass,name,count,offset=0){pass.setPipeline(this.pipelines[name]);pass.setBindGroup(0,this.bind,[offset]);pass.dispatchWorkgroups(count);}
 encodeUpdates(pass,count=this.updates){for(let j=0;j<count;j++){this.dispatch(pass,'forward_backward',Math.ceil(this.config.minibatch/64),j*256);this.dispatch(pass,'gradients',819,j*256);this.dispatch(pass,'adam',Math.ceil(819/64),j*256);}}
 async step(progress=1){
  if(this.fatal)throw Error(this.fatal);const started=performance.now(),d=this.device,c=this.config;
  this.writeConfig(progress);const idx=new Uint32Array(this.b*c.epochs);
  for(let e=0;e<c.epochs;e++){const part=idx.subarray(e*this.b,(e+1)*this.b);for(let i=0;i<this.b;i++)part[i]=i;for(let i=this.b-1;i>0;i--){const j=Math.floor(this.rng()*(i+1)),v=part[i];part[i]=part[j];part[j]=v;}}
  d.queue.writeBuffer(this.indices,0,idx);const update=new Uint32Array(this.updates*64);for(let j=0;j<this.updates;j++){update[j*64]=j*c.minibatch;update[j*64+1]=this.iteration*this.updates+j+1;}d.queue.writeBuffer(this.upd,0,update);
  const enc=d.createCommandEncoder(),pass=enc.beginComputePass(c.timestamp?{timestampWrites:{querySet:this.query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});
  this.dispatch(pass,'rollout',Math.ceil(c.n/32));this.dispatch(pass,'gae',Math.ceil(c.n/64));this.dispatch(pass,'stats_partial',Math.ceil(this.b/256));this.dispatch(pass,'stats_final',1);this.dispatch(pass,'normalize',Math.ceil(this.b/256));this.encodeUpdates(pass);pass.end();
  if(c.timestamp){enc.resolveQuerySet(this.query,0,2,this.queryResult,0);enc.copyBufferToBuffer(this.queryResult,0,this.queryRead,0,16);}
  d.queue.submit([enc.finish()]);await d.queue.onSubmittedWorkDone();const wallMs=performance.now()-started;let gpuMs=null;
  if(c.timestamp){await this.queryRead.mapAsync(GPUMapMode.READ);const times=new BigUint64Array(this.queryRead.getMappedRange());gpuMs=Number(times[1]-times[0])/1e6;this.queryRead.unmap();}
  if(this.fatal)throw Error(this.fatal);this.iteration++;return{iteration:this.iteration,samples:this.iteration*this.b,wallMs,gpuMs,sps:this.b/(wallMs/1000)};
 }
 async read(buffer,count,offset=0){const enc=this.device.createCommandEncoder();enc.copyBufferToBuffer(buffer,offset,this.readBuffer,0,count*4);this.device.queue.submit([enc.finish()]);await this.readBuffer.mapAsync(GPUMapMode.READ,0,count*4);const out=new Float32Array(this.readBuffer.getMappedRange(0,count*4)).slice();this.readBuffer.unmap();return out;}
 async snapshot(){const flat=await this.read(this.w,819);if(!flat.every(Number.isFinite))throw Error('训练产生非有限权重，已停止。');const stats=await this.read(this.roll,4,this.b*16*4);return{flat:Array.from(flat),policy:exportActor(flat),iteration:this.iteration,config:this.config,meanReward:stats[2],terminals:stats[3]};}
 info(){const a=this.adapter.info;return{vendor:a.vendor,architecture:a.architecture,device:a.device,description:a.description,userAgent:navigator.userAgent,worker:typeof document==='undefined',timestampQuery:this.config.timestamp,allocatedBytes:this.allocatedBytes,setupMs:this.setupMs,config:this.config};}
 destroy(){this.query?.destroy();for(const b of this.buffers)b.destroy();this.device.destroy();}
}
