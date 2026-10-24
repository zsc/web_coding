import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {original} from './reference.mjs';
import {hybridInitial} from './hybrid_webgpu.mjs';
const root=new URL('../',import.meta.url);
const read=p=>JSON.parse(fs.readFileSync(new URL(p,root),'utf8'));
const study=read('results/webgpu_ablation.json'),target=read('assets/zhou_target.json');
assert.equal(study.complete,true);assert.equal(study.runs.length,9);
const reference={data:new Uint8ClampedArray(target.W*target.H*4)};
target.pixels.forEach((v,i)=>{reference.data[i*4]=v;});
const checks=[];
for(const [group,r] of Object.entries(study.representatives)){
 const gs=[];
 for(let i=0;i<r.gaussians.length;i+=12){const [x,y,scaleX,scaleY,rotation,opacity]=r.gaussians.slice(i,i+6);gs.push({x,y,scaleX,scaleY,rotation,opacity});}
 const image=original.renderGaussiansToImageData(gs,r.W,r.H),loss=original.calculateLoss(image,reference);
 let different=0;for(let i=0;i<r.W*r.H;i++)if(image.data[i*4]!==r.image[i])different++;
 assert.equal(loss,r.loss);assert.equal(different,0);
 checks.push({group,upstream_mse:loss,browser_mse:r.loss,differing_pixels:different,passed:true});
}
for(const r of study.runs){assert.equal(r.cancelled,false);assert.equal(r.cold_pipeline,false);assert.equal(r.mcmc_done,r.mcmc);assert.equal(r.adam_done,r.adam);assert.equal(r.mode,'fp32');assert.equal(r.N,1600);assert.equal(r.C,8);}
const init=hybridInitial(Uint8Array.from(target.pixels),320,128,1600,8);
const evidence={checks,initial_parameters_sha256:crypto.createHash('sha256').update(new Uint8Array(init.buffer)).digest('hex'),initial_chain_seeds:Array.from({length:8},(_,i)=>42+i*1009),passed:true};
fs.writeFileSync(new URL('results/webgpu_ablation_check.json',root),JSON.stringify(evidence,null,2)+'\n');
console.log(JSON.stringify(evidence,null,2));
