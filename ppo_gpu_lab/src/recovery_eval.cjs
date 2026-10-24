// Reproducible offline recovery assessment and independent physics cross-check.
const fs=require('fs'),path=require('path'),assert=require('node:assert/strict');
const M=require('./live_model.js'),{Mat}=require('./repaired_core.cjs');
const root=path.resolve(__dirname,'..'),read=p=>JSON.parse(fs.readFileSync(path.join(root,p)));
const k=read('results/lqr_design.json').K,p=read('results/desk/v2_physical_seed7.json').policy;
const controllers={ppo10:{limit:10,fn:s=>M.policy(p,s)},lqr10:{limit:10,fn:s=>M.lqr(k,s,10)},lqr40:{limit:40,fn:s=>M.lqr(k,s,40)}};
let rng=20261024;const rand=()=>{rng=(Math.imul(1664525,rng)+1013904223)>>>0;return(rng+.5)/4294967296};
const grid=[];
for(const degrees of[2,5,10,15,20,30]){
  const row={degrees};
  for(const [name,c]of Object.entries(controllers)){
    const cases=[];for(const a of[-degrees,0,degrees])for(const b of[-degrees,0,degrees]){if(a||b)cases.push({angles:[a,b],...M.evaluate(c.fn,[0,0,a*M.RAD,0,b*M.RAD,0],500,c.limit)});}
    row[name]={success:cases.filter(x=>x.success).length,total:8,cases};
  }grid.push(row);
}
const states=Array.from({length:512},()=>[(rand()-.5)*.5,0,(rand()-.5)*20*M.RAD,0,(rand()-.5)*20*M.RAD,0]);
const random={};
for(const [name,c]of Object.entries(controllers)){const cases=states.map(s=>M.evaluate(c.fn,s,500,c.limit));random[name]={success:cases.filter(x=>x.success).length,total:cases.length,fell:cases.filter(x=>x.fell).length};}
// Cross-check the live LDLT implementation against an independently expressed,
// pivoted Gaussian solve (not a separate physical model or external engine).
function independent(s,force){
  const[x,v,t1,w1,t2,w2]=s,c1=Math.cos(t1),c2=Math.cos(t2),c12=Math.cos(t1-t2);
  const A=[[1.2,.15*c1,.05*c2],[.15*c1,.1275,.05*c12],[.05*c2,.05*c12,.0275]];
  const b=[force+.15*Math.sin(t1)*w1*w1+.05*Math.sin(t2)*w2*w2,1.4715*Math.sin(t1)-.05*Math.sin(t1-t2)*w2*w2,.4905*Math.sin(t2)+.05*Math.sin(t1-t2)*w1*w1];
  const [u,a,b2]=Mat.solve(A,b),nv=v+.02*u,nw1=w1+.02*a,nw2=w2+.02*b2;
  return[x+.02*nv,nv,(t1+.02*nw1+Math.PI)%(2*Math.PI)-Math.PI,nw1,(t2+.02*nw2+Math.PI)%(2*Math.PI)-Math.PI,nw2];
}
let error=0;for(let i=0;i<1024;i++){const s=Array.from({length:6},()=>rand()*2-1),f=80*rand()-40;const a=M.step(s,f,40),b=independent(s,f);a.forEach((v,j)=>error=Math.max(error,Math.abs(v-b[j])));}
assert(error<1e-12);
assert(grid.find(x=>x.degrees===10).lqr40.success===8);
assert.equal(M.fallen([0,0,61*M.RAD,0,0,0]),true);
assert.equal(M.evaluate(()=>0,[0,0,2*M.RAD,0,2*M.RAD,0]).success,false);
const demos={};for(const [name,s]of Object.entries({tilt8:[0,0,8*M.RAD,0,8*M.RAD,0],topDrag8:[0,0,0,0,8*M.RAD,0],bottomDrag8:[0,0,8*M.RAD,0,0,0],cartDrag:[.8,0,0,0,0,0],push:[0,.35,0,0,0,0]})){const r=M.evaluate(controllers.lqr40.fn,s,500,40);demos[name]=r;assert(r.success,name);}
const out={model:'physical=True, 20 ms semi-implicit Euler, browser Float64',policy:'v2_physical_seed7, existing weights, no disturbance retraining',success_criterion:'10 s episode; final continuous 1 s with angles <2 deg, angular speeds <0.15 rad/s, cart speed <0.15 m/s; never |angle|>60 deg or |x|>2.4 m',grid,random_rng:'LCG seed 20261024',random_distribution:'512 independent samples, x uniform +/-0.25 m, both angles independent uniform +/-10 degrees, all velocities zero',random,validation:{independent_solver_cases:1024,max_abs:error,demonstrations:demos}};
fs.writeFileSync(path.join(root,'results/recovery_eval.json'),JSON.stringify(out,null,2));
console.log(JSON.stringify({grid:grid.map(r=>({degrees:r.degrees,...Object.fromEntries(Object.keys(controllers).map(k=>[k,r[k].success+'/8']))})),random,physics_max_abs:error,demos:Object.fromEntries(Object.entries(demos).map(([k,v])=>[k,v.recoverySeconds]))},null,2));
