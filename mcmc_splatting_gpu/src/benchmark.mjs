import fs from 'node:fs';import os from 'node:os';import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';
import {rng,initialize,propose,CachedChain,targetFixture} from './core.mjs';import {ReferenceChain} from './reference.mjs';
const root=new URL('../results/',import.meta.url),checks=[];
for(const [W,H,N,seed,T] of [[32,24,12,42,50],[100,50,25,7,50],[25,21,50,9,1],[8,7,2,55,0]]){
  const r=rng(seed),gs=initialize(W,H,N,r),t=targetFixture(W,H);gs[0]={x:0,y:H,scaleX:.1,scaleY:.1,rotation:1.57,opacity:255};
  const ref=new ReferenceChain(gs,W,H,t),fast=new CachedChain(gs,W,H,t);
  let maxLossError=0;for(let i=0;i<500;i++){const p=propose(ref.gs,W,H,r,[20,20,10,10,3,200]);assert.equal(fast.step(p,T),ref.step(p,T),`acceptance ${i}`);maxLossError=Math.max(maxLossError,Math.abs(fast.loss-ref.loss));assert.equal(fast.loss,ref.loss);}
  assert.deepEqual(fast.image(),ref.image());checks.push({W,H,N,seed,T,steps:500,acceptance_match:true,pixel_match:true,maxLossError});
}
console.log('correctness passed',checks.length);
const cases=[];
for(const [W,H,N,iterations] of [[100,50,25,1200],[160,80,64,400],[256,128,128,120]]){
  const t=targetFixture(W,H),initial=initialize(W,H,N,rng(42));const times={reference:[],cached:[]};let output;
  // Both implementations receive identical independent random streams. Warmup excluded.
  for(const C of [ReferenceChain,CachedChain]){const c=new C(initial,W,H,t),r=rng(101);for(let i=0;i<60;i++)c.step(propose(c.gs,W,H,r),50);}
  for(let repeat=0;repeat<3;repeat++)for(const name of (repeat%2?['cached','reference']:['reference','cached'])){
    const C=name==='reference'?ReferenceChain:CachedChain,c=new C(initial,W,H,t),r=rng(101),start=performance.now();
    for(let i=0;i<iterations;i++)c.step(propose(c.gs,W,H,r),50);
    const ms=performance.now()-start;times[name].push(ms);console.log({W,H,N,name,repeat,ms});
    if(name==='cached')output={loss:c.loss,accepted:c.accepted};else if(output){assert.equal(c.loss,output.loss);assert.equal(c.accepted,output.accepted);}
  }
  const median=a=>[...a].sort((a,b)=>a-b)[1],reference_ms=median(times.reference),cached_ms=median(times.cached);
  cases.push({W,H,N,iterations,repeats:3,T:50,seed:42,proposal_seed:101,times,reference_ms,cached_ms,speedup:reference_ms/cached_ms,reference_steps_s:iterations*1000/reference_ms,cached_steps_s:iterations*1000/cached_ms,...output,cache_MiB:N*W*H*8/1048576});
}
const W=100,H=50,N=25,t=targetFixture(W,H),gs=initialize(W,H,N,rng(42)),chain=new CachedChain(gs,W,H,t),r=rng(101),curve=[{iteration:0,loss:chain.loss}],initial=Array.from(chain.image());
const start=performance.now();let best={loss:chain.loss,image:initial,iteration:0};
for(let i=0;i<10000;i++){chain.step(propose(chain.gs,W,H,r),50);if(chain.loss<best.loss)best={loss:chain.loss,image:Array.from(chain.image()),iteration:i+1};if((i+1)%100===0)curve.push({iteration:i+1,loss:chain.loss});}
const demo={W,H,N,iterations:10000,T:50,ms:performance.now()-start,target:Array.from(t),initial,final:Array.from(chain.image()),loss:chain.loss,best,curve,accepted:chain.accepted};
const data={measured_at:new Date().toISOString(),machine:{cpu:os.cpus()[0].model,platform:os.platform(),arch:os.arch(),node:process.version,memory_bytes:os.totalmem()},method:'single process; same JS realm; no DOM/drawing/rAF; 60 warmup steps per implementation; 3 interleaved timed runs; initialization excluded; full-image quantized MSE',checks,cases,demo};
fs.writeFileSync(new URL('local_benchmark.json',root),JSON.stringify(data,null,2));console.log('saved local_benchmark.json');
