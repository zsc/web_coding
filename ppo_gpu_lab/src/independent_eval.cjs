// Independent scalar Float64 JS rollout of the exported GPU policy.
const fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'..');
const {Mat,DoubleInvertedPendulumEnv:Env}=require('./repaired_core.cjs');
let seed=811;Math.random=()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return(seed+.5)/4294967296};
function forward(policy,s){for(let i=0;i<=4;i+=2){s=policy[i+'.weight'].map((row,j)=>row.reduce((sum,w,k)=>sum+w*s[k],policy[i+'.bias'][j]));if(i<4)s=s.map(x=>Math.max(x,0));}return Math.max(-1,Math.min(1,s[0]));}
function physicalStep(force){const [x,v,t1,w1,t2,w2]=this.state,c1=Math.cos(t1),c2=Math.cos(t2),c12=Math.cos(t1-t2),s1=Math.sin(t1),s2=Math.sin(t2),s12=Math.sin(t1-t2);
const M=[[1.2,.15*c1,.05*c2],[.15*c1,.1275,.05*c12],[.05*c2,.05*c12,.0275]];
const N=[force+.15*s1*w1*w1+.05*s2*w2*w2,1.4715*s1-.05*s12*w2*w2,.4905*s2+.05*s12*w1*w1];
const [a,b,c]=Mat.solve(M,N),nv=v+.02*a,nw1=w1+.02*b,nw2=w2+.02*c;
this.state=[x+.02*nv,nv,(t1+.02*nw1+Math.PI)%(2*Math.PI)-Math.PI,nw1,(t2+.02*nw2+Math.PI)%(2*Math.PI)-Math.PI,nw2];}
const results=[];
for(const name of ['graph_scaled_seed7','graph_scaled_seed11','graph_scaled_seed23','physical_seed7','v2_physical_seed7','v2_physical_seed11','v2_physical_seed23']){
const folder=fs.existsSync(path.join(root,'results/desk'))?'results/desk':'results';
if(!fs.existsSync(path.join(root,folder,name+'.json')))continue;
const run=JSON.parse(fs.readFileSync(path.join(root,folder,name+'.json')));
const env=new Env();if(run.config.physical)env._physics_step=physicalStep;
let total=0,length=0,success=0;
seed=811;
for(let ep=0;ep<128;ep++){let s=env.reset(),ret=0,n=0,done=false;while(n<1000&&!done){const r=env.step(forward(run.policy,s));s=r.observation;done=r.done;ret+=r.reward;n++;}total+=ret;length+=n;success+=!done;}
results.push({name,independent_engine:'scalar Float64 JavaScript, pivoted Gaussian solve',reset_rng:'LCG seed 811 (independent of PyTorch)',episodes:128,mean_return:total/128,mean_length:length/128,survival_1000:success/128});
}
fs.writeFileSync(path.join(root,'results/independent_eval.json'),JSON.stringify(results,null,2));console.log(JSON.stringify(results,null,2));
