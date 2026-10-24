import fs from 'node:fs';
import assert from 'node:assert/strict';
import {original} from './reference.mjs';
const root=new URL('../',import.meta.url),folder=new URL('results/desk_muon_study/',root);
const study=JSON.parse(fs.readFileSync(new URL('study.json',folder),'utf8'));
const target=JSON.parse(fs.readFileSync(new URL('assets/zhou_target.json',root),'utf8'));
assert.ok(study.complete);assert.equal(study.runs.length,29);
const tar={data:new Uint8ClampedArray(target.W*target.H*4)};target.pixels.forEach((v,i)=>tar.data[i*4]=v);
const checks=[];
for(const run of study.runs){
 const r=JSON.parse(fs.readFileSync(new URL(run.file,folder),'utf8'));
 assert.equal(r.N,1600);assert.equal(r.chains,8);assert.equal(r.seed,run.seed);
 assert.equal(r.steps,run.steps);assert.equal(r.mode,run.mode);assert.ok(Math.abs(r.lr-run.lr)<1e-8);
 assert.ok(r.gaussians.flat().every(Number.isFinite));
 const gs=r.gaussians.map(([x,y,scaleX,scaleY,rotation,opacity])=>({x,y,scaleX,scaleY,rotation,opacity}));
 const image=original.renderGaussiansToImageData(gs,r.W,r.H),mse=original.calculateLoss(image,tar);
 const pgm=fs.readFileSync(new URL(run.file.replace('.json','.pgm'),folder));
 const header=Buffer.from(`P5\n${r.W} ${r.H}\n255\n`);assert.ok(pgm.subarray(0,header.length).equals(header));
 let differing=0;for(let p=0;p<r.W*r.H;p++)if(image.data[p*4]!==pgm[header.length+p])differing++;
 assert.equal(differing,0);assert.ok(Math.abs(mse-r.loss)<1e-9);
 checks.push({file:run.file,upstream_mse:mse,reported_mse:r.loss,differing_pixels:differing,passed:true});
}
fs.writeFileSync(new URL('renderer_check.json',folder),JSON.stringify({passed:true,checked:checks.length,checks},null,2)+'\n');console.log(`Passed ${checks.length} original-renderer checks.`);
