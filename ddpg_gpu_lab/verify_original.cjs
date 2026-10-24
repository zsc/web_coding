// Evaluate exported actors in the exact original JavaScript environment.
// Four corners of the original initial-state range; FP64, no training.
const fs=require('fs'),{InvertedPendulumEnv}=require('./core_original.cjs');
const file=process.argv[2],horizon=+(process.argv[3]||10000),out=process.argv[4]||'results/verification_original.json';
const d=JSON.parse(fs.readFileSync(file)),td3=d.config.algo==='td3';
const scale=td3?[1/2.4,.5,15/Math.PI,.5]:[1,1,1,1];
const params={gravity:9.8,masscart:1,masspole:.1,length:.5,tau:.02,action_bound:10,theta_threshold_radians:Math.PI/15,x_threshold:2.4};
function rawForward(w,s,logits=false){let a=s.map((v,i)=>v*scale[i]);for(let l=0;l<3;l++){let z=w[2*l].map((row,j)=>{let v=w[2*l+1][j];for(let k=0;k<a.length;k++)v+=row[k]*a[k];return l===2?(logits?v:10*Math.tanh(v)):Math.max(0,v)});a=z;}return a[0];}
function forward(w,s){if(!d.config.symmetry)return rawForward(w,s);const pre=d.config.symmetry_mode==='pre_tanh_logit_odd';let z=.5*(rawForward(w,s,pre)-rawForward(w,s.map(v=>-v),pre));return pre?10*Math.tanh(z):z;}
const starts=[[-.01,-.01],[-.01,.01],[.01,-.01],[.01,.01]],results=[],start=performance.now();
for(const [seed,w] of Object.entries(d.actors)){
 let lengths=[],survived=0;
 for(const [theta,td] of starts){let e=new InvertedPendulumEnv(params);e.state=[0,0,theta,td];let n=0,done=false;
  for(;n<horizon&&!done;n++){let t=e.step([forward(w,e.state)]);done=t.done;}
  lengths.push(n);survived+=!done;
 }
 results.push({seed:+seed,lengths,survived});
}
const result={source:file,backend:'Original unmodified JavaScript environment + FP64 actor inference',horizon,initial_states:starts.map(v=>[0,0,...v]),passed:results.filter(x=>x.survived===4).length,seeds:results.length,seconds:(performance.now()-start)/1000,results};
fs.writeFileSync(out,JSON.stringify(result,null,2));console.log(JSON.stringify({...result,results:undefined}));
