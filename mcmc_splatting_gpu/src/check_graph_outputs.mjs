import fs from 'node:fs';
import assert from 'node:assert/strict';
import {original} from './reference.mjs';
const root=new URL('../',import.meta.url),read=p=>JSON.parse(fs.readFileSync(new URL(p,root),'utf8'));
const pilotChecks=[];
for(const mode of [0]){
 const a=read(`results/desk_graph_pilot/check_m${mode}_b0_t1_s8_n32.json`),b=read(`results/desk_graph_pilot/check_m${mode}_b3_t1_s8_n32.json`);
 const max=Math.max(...a.gaussians.flat().map((v,i)=>Math.abs(v-b.gaussians.flat()[i])));assert.ok(max<1e-4);assert.equal(b.graph_iterations,6);
 const c=read(`results/desk_graph_warp/m${mode}_w0_b0_s8.json`),d=read(`results/desk_graph_warp/m${mode}_w1_b0_s8.json`);
 const warpMax=Math.max(...c.gaussians.flat().map((v,i)=>Math.abs(v-d.gaussians.flat()[i])));assert.ok(warpMax<1e-3);
 pilotChecks.push({mode,graph_max_parameter_difference:max,expected_graph_iterations:6,warp_max_parameter_difference:warpMax,passed:true});
}
const s=read('results/desk_graph_study/study.json'),t=read('assets/zhou_target.json');assert.ok(s.complete);assert.equal(s.runs.length,49);
const tar={data:new Uint8ClampedArray(t.W*t.H*4)};t.pixels.forEach((v,i)=>tar.data[i*4]=v);
const checks=[];
for(const run of s.runs.filter(r=>r.mode===0)){
 const r=read('results/desk_graph_study/'+run.file);assert.equal(r.steps,run.steps);assert.equal(r.seed,run.seed);assert.equal(r.graph_batch,Math.min(run.graph,r.steps));assert.equal(r.warp_roi,run.warp);
 assert.equal(r.graph_iterations,run.graph?Math.floor(r.steps/r.graph_batch)*r.graph_batch:0);assert.ok(r.gaussians.flat().every(Number.isFinite));
 const gs=r.gaussians.map(([x,y,scaleX,scaleY,rotation,opacity])=>({x,y,scaleX,scaleY,rotation,opacity}));
 const image=original.renderGaussiansToImageData(gs,r.W,r.H),mse=original.calculateLoss(image,tar);
 const pgm=fs.readFileSync(new URL('results/desk_graph_study/'+run.file.replace('.json','.pgm'),root));const header=Buffer.from(`P5\n${r.W} ${r.H}\n255\n`);assert.ok(pgm.subarray(0,header.length).equals(header));let differing=0;
 for(let p=0;p<r.W*r.H;p++)if(image.data[p*4]!==pgm[header.length+p])differing++;
 assert.equal(differing,0);assert.ok(Math.abs(mse-r.loss)<1e-9);
 checks.push({file:run.file,original_mse:mse,reported_mse:r.loss,differing_pixels:differing,graph_iterations:r.graph_iterations,passed:true});
}
fs.writeFileSync(new URL('results/desk_graph_study/checks.json',root),JSON.stringify({passed:true,checked:checks.length,pilot_checks:pilotChecks,checks},null,2)+'\n');console.log(`Passed ${checks.length} full original-renderer checks, graph counters, and 2 paired short-run comparisons.`);
