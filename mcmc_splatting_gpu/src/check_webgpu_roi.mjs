import fs from 'node:fs';import assert from 'node:assert/strict';import {original} from './reference.mjs';
const root=new URL('../',import.meta.url),r=JSON.parse(fs.readFileSync(new URL('results/webgpu_roi_study.json',root),'utf8'));assert.ok(r.complete);assert.equal(r.runs.length,13);const checks=[];
for(const [name,rep] of Object.entries(r.representatives)){
 const gs=[];for(let j=0;j<rep.gaussians.length;j+=12){const [x,y,scaleX,scaleY,rotation,opacity]=rep.gaussians.slice(j,j+6);gs.push({x,y,scaleX,scaleY,rotation,opacity});}
 const image=original.renderGaussiansToImageData(gs,rep.W,rep.H),tar={data:new Uint8ClampedArray(rep.W*rep.H*4)};rep.target.forEach((v,i)=>tar.data[i*4]=v);const loss=original.calculateLoss(image,tar);let different=0;for(let p=0;p<rep.image.length;p++)if(rep.image[p]!==image.data[p*4])different++;
 assert.equal(different,0);assert.ok(Math.abs(loss-rep.loss)<1e-9);checks.push({name,original_mse:loss,reported_mse:rep.loss,differing_pixels:different,passed:true});
}
assert.ok(r.runs.filter(x=>x.parallel_check).every(x=>x.parallel_check.passed));fs.writeFileSync(new URL('results/webgpu_roi_checks.json',root),JSON.stringify({passed:true,checked:checks.length,checks},null,2)+'\n');console.log('Passed 4 original-renderer checks and worker partition checks.');
