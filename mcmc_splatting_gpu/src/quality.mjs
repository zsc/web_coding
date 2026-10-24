import fs from 'node:fs';import assert from 'node:assert/strict';import {performance} from 'node:perf_hooks';
import {CachedChain,initialize,propose,rng,targetFixture} from './core.mjs';import {ReferenceChain} from './reference.mjs';
const W=100,H=50,t=targetFixture(W,H),runs=[];
for(const [N,steps,mode] of [[25,10000,'fixed'],[25,10000,'annealed'],[100,100000,'annealed']]){
  const c=new CachedChain(initialize(W,H,N,rng(42)),W,H,t),r=rng(101),curve=[{iteration:0,loss:c.loss}];let best={loss:c.loss,image:Array.from(c.image()),iteration:0},maxRebuildDrift=0,checkpoints=0;
  const start=performance.now();for(let i=0;i<steps;i++){
    const T=mode==='fixed'?50:50*Math.pow(.05/50,i/Math.max(1,steps-1));
    c.step(propose(c.gs,W,H,r),T);
    if(c.loss<best.loss)best={loss:c.loss,image:Array.from(c.image()),iteration:i+1};
    if((i+1)%Math.max(100,Math.floor(steps/100))===0)curve.push({iteration:i+1,loss:c.loss});
    if((i+1)%5000===0){const ref=new ReferenceChain(c.gs,W,H,t);assert.equal(c.loss,ref.loss);assert.deepEqual(c.image(),ref.image());const sum=c.sum.slice();c.rebuild();for(let p=0;p<W*H;p++)maxRebuildDrift=Math.max(maxRebuildDrift,Math.abs(sum[p]-c.sum[p]));checkpoints++;}
  }
  runs.push({W,H,N,steps,mode,initial_T:50,final_T:mode==='fixed'?50:.05,ms_including_checkpoints:performance.now()-start,loss:c.loss,image:Array.from(c.image()),accepted:c.accepted,best,curve,maxRebuildDrift,checkpoints});
  console.log({N,steps,mode,loss:c.loss,best:best.loss,ms:runs.at(-1).ms_including_checkpoints});
}
fs.writeFileSync(new URL('../results/quality.json',import.meta.url),JSON.stringify({runs},null,2));
