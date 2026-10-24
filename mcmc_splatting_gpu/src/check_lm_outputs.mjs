import fs from 'node:fs';
import assert from 'node:assert/strict';
import {original} from './reference.mjs';
const root=new URL('../',import.meta.url);
const read=p=>JSON.parse(fs.readFileSync(new URL(p,root),'utf8'));
const study=read('results/desk_lm_study/study.json'),target=read('assets/zhou_target.json');
assert.ok(study.complete);assert.equal(study.runs.length,19);
const tar={data:new Uint8ClampedArray(target.W*target.H*4)};
target.pixels.forEach((v,i)=>tar.data[i*4]=v);
const checks=[];
for(const run of study.runs){
 const r=read('results/desk_lm_study/'+run.file);
 const gs=r.gaussians.map(([x,y,scaleX,scaleY,rotation,opacity])=>({x,y,scaleX,scaleY,rotation,opacity}));
 const image=original.renderGaussiansToImageData(gs,r.W,r.H),mse=original.calculateLoss(image,tar);
 const pgm=fs.readFileSync(new URL('results/desk_lm_study/'+run.file.replace('.json','.pgm'),root));
 const header=Buffer.from(`P5\n${r.W} ${r.H}\n255\n`);assert.ok(pgm.subarray(0,header.length).equals(header));
 let different=0;for(let p=0;p<r.W*r.H;p++)if(image.data[p*4]!==pgm[header.length+p])different++;
 assert.equal(different,0);assert.ok(Math.abs(mse-r.loss)<1e-9);
 assert.equal(r.monotonic_smooth_loss,true);
 for(let i=1;i<r.smooth_curve.length;i++)for(let c=0;c<r.chains;c++)assert.ok(r.smooth_curve[i].losses[c]<=r.smooth_curve[i-1].losses[c]+1e-9);
 for(const s of r.lm_states)assert.equal(s.accepted+s.rejected,r.lm_steps);
 checks.push({file:run.file,upstream_mse:mse,reported_mse:r.loss,differing_pixels:different,monotonic_curve:true,passed:true});
}
const result={checked:checks.length,passed:true,checks};
fs.writeFileSync(new URL('results/desk_lm_study/renderer_check.json',root),JSON.stringify(result,null,2)+'\n');console.log(`Passed ${checks.length} original-renderer and LM-acceptance checks.`);
