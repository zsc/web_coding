// Fresh post-freeze audit. Does not choose hyperparameters or checkpoints.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {assessSuite,policyFunction} from './webgpu_eval.mjs';
const require=createRequire(import.meta.url),model=require('./live_model.js');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const read=f=>JSON.parse(fs.readFileSync(path.join(root,'results',f)));
const initial=read('webgpu_initial.json'),seed=271828,count=1024;
const options={seed,count},runs=[];
for(const trainingSeed of [71,79,83]){
 const r=read(`webgpu_train_seed${trainingSeed}.json`);
 const result=assessSuite(r.best.flat,40,model,options),control=policyFunction(r.best.flat,40);
 const examples={both8:[0,0,8*model.RAD,0,8*model.RAD,0],lower8:[0,0,8*model.RAD,0,0,0],upper8:[0,0,0,0,8*model.RAD,0],cart04:[.4,0,0,0,0,0],push035:[0,.35,0,0,0,0]};
 runs.push({trainingSeed,checkpoint:r.best.iteration,totalIterations:r.completedIterations,result,interactionExamples:Object.fromEntries(Object.entries(examples).map(([k,s])=>[k,model.evaluate(control,s,500,40)]))});
}
const output={note:'Fresh post-freeze audit seed 271828, 1024 episodes per condition, never used to select checkpoints. Independent from WGSL arithmetic, same intended physical equations; not a third-party physics engine.',options,original10N:assessSuite(initial,10,model,options),initial40N:assessSuite(initial,40,model,options),runs};
fs.writeFileSync(path.join(root,'results/webgpu_independent_eval.json'),JSON.stringify(output,null,2));
console.log(JSON.stringify({original10N:output.original10N.angle10.success,runs:runs.map(x=>({seed:x.trainingSeed,checkpoint:x.checkpoint,angle10:x.result.angle10.success,pushes:x.result.pushes.success,examples:Object.fromEntries(Object.entries(x.interactionExamples).map(([k,v])=>[k,v.success]))}))},null,2));
